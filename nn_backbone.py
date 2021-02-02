import tensorflow as tf
import numpy as np
from config import path_to_vg, path_to_ukiyoe

vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
gen_vg = tf.keras.models.load_model(path_to_vg)
gen_ukiyoe = tf.keras.models.load_model(path_to_ukiyoe)


def clip_image(img):
    """
      Приводим изображение к диапазону [0, 1]
    """
    return tf.clip_by_value(img, clip_value_min=0.0, clip_value_max=1.0)


def get_model(layer_names):
    """
      Функция для создания модели
    """
    vgg.trainable = False
    outputs = [vgg.get_layer(layer).output for layer in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model


def gram(x):
    """
     Функция для вычисления матрицы Грама
    """
    matrix = tf.linalg.einsum('bijc,bijd->bcd', x, x)
    in_shape = tf.shape(x)
    dim = tf.cast(in_shape[1] * in_shape[2] * in_shape[3], tf.float32)
    return matrix / dim


class StyleAndContentExtractor:
    """
     Экстрактор для фич стиля и контента
    """

    def __init__(self, style_layers, content_layers):
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.vgg_outputs_model = get_model(style_layers + content_layers)
        self.vgg_outputs_model.trainable = False

    def __call__(self, inputs):
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(
            inputs * 255.)  # VGG препроцессинг [0, 1] -> [0, 255], также нормализация изображения
        outputs = self.vgg_outputs_model(preprocessed_input)
        features_dict = {}  # Словарь, содержащий слои стиля и контента

        style_outputs = [gram(style_output)
                         for style_output in outputs[:len(self.style_layers)]]
        content_outputs = outputs[len(self.style_layers):]

        features_dict["style"] = {layer: out for layer, out in zip(self.style_layers, style_outputs)}
        features_dict["content"] = {layer: out for layer, out in zip(self.content_layers, content_outputs)}

        return features_dict


# Задаем слои стиля и контента
style_layers = ['block1_conv1',
                'block1_conv2',
                'block2_conv1',
                'block2_conv2',
                'block3_conv2',
                'block3_conv3',
                'block4_conv2',
                'block4_conv3',
                'block5_conv2']
content_layers = ['block3_conv1', 'block4_conv1']

# Создаем объект класса StyleAndContentExtractor
extractor = StyleAndContentExtractor(style_layers=style_layers, content_layers=content_layers)


def total_loss(img, style_targets, content_targets, style_weight=300.0, content_weight=1, tv_weight=0.5):
    """
      Функция для подсчета полных потерь
    """
    style_loss = None
    content_loss = None
    style_and_content_features = extractor(img)
    style_features = style_and_content_features['style']
    content_features = style_and_content_features['content']
    style_loss = tf.add_n([tf.keras.losses.MeanSquaredError()(style_features[layer], style_targets[layer])
                           for layer in style_features.keys()])
    style_loss *= 1. / len(style_features)

    content_loss = tf.add_n([tf.keras.losses.MeanSquaredError()(content_features[layer], content_targets[layer])
                             for layer in content_features.keys()])
    content_loss *= 1. / len(content_features)
    loss = style_weight * style_loss + content_weight * content_loss + tv_weight * tf.image.total_variation(img)
    return loss


def img_preprocess(img, GAN=0):
    """
      Функция для препроцессинга изображения в виде байтовой последовательности,
      полученного от бота
    """
    dim = 256  # Размерность всех выходных картинок 256*256
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    if GAN:  # Для GAN трансформации нам нужно сделать преобразование [0, 1] -> [-0.5, 0.5]
        img = img - 0.5
    shape = np.array(img.shape)[:-1]
    # Осуществляем обработку "resize and center crop"
    min_dim = min(shape)
    scale = dim / min_dim
    new_shape = tuple((shape * scale).astype("int32"))
    img = tf.image.resize(img, new_shape, preserve_aspect_ratio=True)
    img = tf.image.resize_with_crop_or_pad(img, dim, dim)
    img = img[tf.newaxis, :]  # Добавляем размерность батча
    return img


def get_image(content_img, style_img):
    """
      Функция по полученным от бота изображениям стиля и контента
      делает перенос стиля, возвращает стилизованное изображение
    """

    def loss(img):
        return total_loss(img, style_targets, content_targets)

    def step(img, loss_func, optimizer):
        """
          Один шаг тренировочного цикла
        """

        with tf.GradientTape() as tape:
            loss = loss_func(img)
        grad = tape.gradient(loss, img)  # dloss/dImage
        optimizer.apply_gradients([(grad, img)])
        img.assign(clip_image(img))
        return loss.numpy()

    content_img = tf.Variable(img_preprocess(content_img))
    style_img = img_preprocess(style_img)
    content_targets = extractor(content_img)['content']
    style_targets = extractor(style_img)['style']
    opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    for n in range(500):  # Train loop
        step(content_img, loss_func=loss, optimizer=opt)

    img = tf.squeeze(content_img, axis=0)
    return img


def get_gan_image(img, GAN=1):
    """
      Функция применяет к изображению одну из предобученных GAN-моделей,
      возвращает стилизованное изображение
    """
    img = img_preprocess(img, GAN=GAN)
    if GAN == 1:
        img = gen_ukiyoe(img)
    else:
        img = gen_vg(img)
    img = img * 0.5 + 0.5  # [-0.5, 0.5] -> [0, 1]
    img = tf.squeeze(img, axis=0)
    return img
