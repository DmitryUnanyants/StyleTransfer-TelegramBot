# Телеграм-бот для переноса стиля
Данный проект представляет собой Телеграм-бота для переноса стиля с одного изображения на другое. <br><br>
Также поддерживается быстрый перенос стиля с двух предобученных GAN-моделей: стиль Ван Гога и японский стиль Укиё-э.<br><br>
***Запуск:*** python3 bot.py <br> <br>
Перенос стиля осуществляется с помощью нейросетей, соответствующая логика в файле nn_backbone.py <br><br>
Все подсказки по использованию бота вызываются командой /help <br><br>
Запускать желательно на машине с GPU c видеопамятью >= 8GB, тогда процесс переноса стиля в режиме user-style будет осуществляться намного быстрее.<br><br>
В файле config.py надо указать пути к предобученным GAN-моделям (подробнее в комментариях в файле config.py), а также токен бота (можно получить у @BotFather).<br><br>
Модели GAN были обучены в соответсвии с данным туториалом: https://www.tensorflow.org/tutorials/generative/cyclegan <br><br>
**Внешний вид бота:**<br><br>
![Скриншот окна Телеграм](screenshot.jpg)<br>
