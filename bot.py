from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
from io import BytesIO
from config import token
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from nn_backbone import get_gan_image
from nn_backbone import get_image
from tensorflow.keras.preprocessing.image import array_to_img


bot = Bot(token=token)
dp = Dispatcher(bot, storage=MemoryStorage())
titles = {1: "японском стиле Укиё-э", 2: "стиле художника Ван Гога"}


class BotStates(StatesGroup):
    """
      Класс - машина состояний для бота. Content - состояние обработки фото контента,
      style - состояние обработки фото стиля
    """
    content = State()
    style = State()


def prepare_to_send(img):
    """
      Приводим изображение к байтовому буферу, который бот непосредственно
      отправляет в чат
    """
    BIO = BytesIO()
    BIO.seek(0)
    BIO.name = "im.png"
    img.save(BIO)
    BIO.seek(0)
    return BIO


@dp.message_handler(commands=['start'], state="*")
async def start_msg(msg, state):
    keyboard = types.ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    button_start = types.KeyboardButton(text="/start")
    button_help = types.KeyboardButton(text="/help")
    button_nst = types.KeyboardButton(text="/user_style")
    button_gan1 = types.KeyboardButton(text="/Ukiyo-e")
    button_gan2 = types.KeyboardButton(text="/VanGogh")
    keyboard.add(button_start, button_help)
    keyboard.row(button_nst, button_gan1, button_gan2)
    await BotStates.content.set()
    await state.update_data(transfer_method=0)
    await bot.send_message(msg.chat.id, f'Привет, {msg.from_user.first_name}! Это бот для переноса стиля\
 с одного изображения (style), на другое (content). Для более подробной информации нажмите /help.',
                           reply_markup=keyboard)


@dp.message_handler(commands=['help'], state="*")
async def help_msg(msg):
    await bot.send_message(msg.chat.id, "Команды чат-бота:\n/start - начало работы и общий сброс\n/help\
 - помощь\n/user_style - стиль пользователя (это режим по умолчанию). Сначала надо загрузить изображение content,\
 затем изображение style\n/Ukiyoe - к изображению, загруженному пользователем, будет применен японский стиль Укиё-э\
\n/VanGogh - к изображению, загруженному пользователем, будет применен стиль художника Ван Гога.\n\n Методы стилизации Ukiyoe\
 и VanGogh (основаны на сетях GAN) применяются быстро, метод user_style может занять некоторое время (до минуты).")


@dp.message_handler(commands=['user_style'], state="*")
async def send_msg(msg, state):
    await bot.send_message(msg.chat.id, "Загрузите изображение content, к которому будет применяться стилизация\
, обычным способом.")
    await BotStates.content.set()
    await state.update_data(transfer_method=0)


@dp.message_handler(commands=['Ukiyoe'], state="*")
async def send_msg(msg, state):
    await bot.send_message(msg.chat.id, "Загрузите изображение content, к которому будет применен японский стиль Укиё-э\
, обычным  способом.")
    await BotStates.content.set()
    await state.update_data(transfer_method=1)


@dp.message_handler(commands=['VanGogh'], state="*")
async def send_msg(msg, state):
    await bot.send_message(msg.chat.id, "Загрузите изображение content, к которому будет применен стиль художника Ван Гога\
, обычным способом")
    await BotStates.content.set()
    await state.update_data(transfer_method=2)


@dp.message_handler(content_types=["photo", "document"], state=BotStates.content)
async def send_photo(msg, state):
    if msg.content_type == "photo":
        pic = await bot.download_file_by_id(msg.photo[0].file_id)
        pic = pic.read()
    elif msg.document.mime_type.split("/")[0] == "image":
        pic = await bot.download_file_by_id(msg.document.file_id)
        pic = pic.read()
    else:
        await bot.send_message(msg.chat.id, "Это не изображение, я жду изображений:)")
        return 0
    user_data = await state.get_data()
    mode = user_data["transfer_method"]
    if mode == 0:
        await state.update_data(content_pic=pic)
        await bot.send_message(msg.chat.id,
                               "Отлично, изображение content получено! Теперь загрузите изображение style.")
        await BotStates.style.set()

    else:
        final_image = get_gan_image(pic, GAN=mode)
        final_image = array_to_img(final_image)
        await bot.send_message(msg.chat.id, f"Ваше изображение в {titles[mode]}:")
        await bot.send_photo(msg.from_user.id, prepare_to_send(final_image))


@dp.message_handler(content_types=["photo", "document"], state=BotStates.style)
async def send_photo(msg, state):
    if msg.content_type == "photo":
        pic = await bot.download_file_by_id(msg.photo[0].file_id)
        pic = pic.read()
    elif msg.document.mime_type.split("/")[0] == "image":
        pic = await bot.download_file_by_id(msg.document.file_id)
        pic = pic.read()
    else:
        await bot.send_message(msg.chat.id, "Это не изображение, я жду изображений:)")
        return 0

    user_data = await state.get_data()
    c_im = user_data["content_pic"]
    s_im = pic
    await bot.send_message(msg.chat.id, "Процесс переноса стиля запущен, ожидайте около минуты.")
    final_image = get_image(c_im, s_im)
    final_image = array_to_img(final_image)
    await bot.send_message(msg.chat.id, "Ваше стилизованное изображение:")
    await bot.send_photo(msg.from_user.id, prepare_to_send(final_image))
    await BotStates.content.set()


@dp.message_handler(content_types=types.ContentTypes.ANY, state="*")
async def send_text(msg):
    await bot.send_message(msg.chat.id, "Я не понимаю, что должен делать")


if __name__ == '__main__':
    executor.start_polling(dp)
