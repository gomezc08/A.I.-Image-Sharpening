{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Training",
      "provenance": [],
      "collapsed_sections": [],
      "machine_shape": "hm",
      "authorship_tag": "ABX9TyM2C1fcWB8u9YaxxagMsz1O",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU",
    "gpuClass": "standard"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/gomezc08/A.I.-Image-Sharpening/blob/main/Training.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "#A.I. Image Sharpening"
      ],
      "metadata": {
        "id": "zHujXsJiyQnN"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "metadata": {
        "id": "MwGrfWt8es0f"
      },
      "outputs": [],
      "source": [
        "import tensorflow as tf\n",
        "import tqdm\n",
        "import os\n",
        "import time\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import cv2\n",
        "import matplotlib.pyplot as plt\n",
        "import keras.backend as K\n",
        "from PIL import Image, ImageFilter\n",
        "from matplotlib import image \n",
        "from IPython import display\n",
        "from shutil import copyfile\n",
        "from tensorflow.python.keras.layers import Layer, InputSpec \n",
        "from keras.applications.vgg16 import VGG16\n",
        "from re import I"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Google Colab Set-up\n",
        "Little side note: I used google colab, which may required a change in some syntax features, such as loading pictures in files"
      ],
      "metadata": {
        "id": "xQ91XypSOiO1"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/gdrive')\n",
        "%cd /gdrive"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "r79LEG3jfh0R",
        "outputId": "3760eb4d-9f3c-4a3e-f71f-bf03e8e4d647"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Drive already mounted at /gdrive; to attempt to forcibly remount, call drive.mount(\"/gdrive\", force_remount=True).\n",
            "/gdrive\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Google Drive Directories"
      ],
      "metadata": {
        "id": "aVHaMImFO1dx"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "TRAIN_DIR = r'/gdrive/MyDrive/GOPRO_Large/train'\n",
        "TRAIN_DIR2 = r'/gdrive/MyDrive/Pics/train'\n",
        "TEST_DIR = r'/gdrive/MyDrive/GOPRO_Large/test'"
      ],
      "metadata": {
        "id": "q_9dGCAr8lbg"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ngf = 64 \n",
        "ndf = 64\n",
        "input_nc = 3\n",
        "output_nc = 3\n",
        "input_shape_generator = (256, 256, input_nc)\n",
        "n_blocks_gen = 9\n",
        "image_shape = (256,256, 3)\n",
        "input_shape_discriminator = (256, 256, output_nc)\n",
        "EPOCHS = 2    #can change to 50 laters\n",
        "noise_dim = 100\n",
        "num_examples_to_generate = 16\n",
        "BUFFER_SIZE = 60000\n",
        "BATCH_SIZE = 256  "
      ],
      "metadata": {
        "id": "KaIDuX3hevS5"
      },
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def loadData():\n",
        "    trainDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)\n",
        "    trainGenerator = trainDataGenerator.flow_from_directory (\n",
        "        TRAIN_DIR,\n",
        "        target_size = (256,256),\n",
        "        class_mode = 'categorical'\n",
        "    )\n",
        "\n",
        "    testDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)\n",
        "    testGenerator = testDataGenerator.flow_from_directory (\n",
        "        TEST_DIR,\n",
        "        target_size = (256,256),\n",
        "        class_mode = 'categorical'\n",
        "    )\n",
        "    train_y = trainGenerator.class_indices\n",
        "    test_y = testGenerator.class_indices\n",
        "\n",
        "    return trainGenerator, train_y, testGenerator, test_y"
      ],
      "metadata": {
        "id": "eG4QVE3oe0K6"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_x, train_y, test_x, test_y = loadData()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "1UvFA0FptLic",
        "outputId": "bbd96f5d-d2a1-4fcc-e330-9cdd16ce94ef"
      },
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Found 983 images belonging to 5 classes.\n",
            "Found 1583 images belonging to 5 classes.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Blur Images in File"
      ],
      "metadata": {
        "id": "owOEqhr-jONv"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def blur(path, testOrTrain):\n",
        "    count = 1\n",
        "    for f in os.listdir(path):\n",
        "        c = path + \"/\" + f\n",
        "        pic = Image.open(c)\n",
        "        pic = pic.filter(ImageFilter.GaussianBlur)    #BLURRY PHOTO\n",
        "        if testOrTrain == 'train':\n",
        "            os.chdir(TRAIN_DIR + '/train')\n",
        "        else:\n",
        "            os.chdir(TEST_DIR + '/test')\n",
        "        \n",
        "        pic.save('savedImage' + str(count) + '.png')\n",
        "        count += 1"
      ],
      "metadata": {
        "id": "Sg2RN_w0jMP6"
      },
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Creating new Layer for Model"
      ],
      "metadata": {
        "id": "M2duRkODOggk"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#found this layer on tensorflow website (link provided below)\n",
        "#but basiicaally I wanted to pad down a tensor\n",
        "\n",
        "class ReflectionPadding2D(Layer):\n",
        "    def __init__(self, padding=(1, 1), **kwargs):\n",
        "        self.padding = tuple(padding)\n",
        "        self.input_spec = [InputSpec(ndim=4)]\n",
        "        super(ReflectionPadding2D, self).__init__(**kwargs)\n",
        "\n",
        "    def get_output_shape_for(self, s):\n",
        "        \"\"\" If you are using \"channels_last\" configuration\"\"\"\n",
        "        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])\n",
        "\n",
        "    def call(self, x, mask=None):\n",
        "        w_pad,h_pad = self.padding\n",
        "        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')"
      ],
      "metadata": {
        "id": "t9wXXMs2e84k"
      },
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def resNetBlock(input, filters, kernel_size=(3,3), strides=(1,1), use_dropout=False):\n",
        "    x = ReflectionPadding2D((1,1))(input)\n",
        "    x = tf.keras.layers.Conv2D(filters=filters,\n",
        "               kernel_size=kernel_size,\n",
        "               strides=strides,)(x)\n",
        "    x = tf.keras.layers.BatchNormalization()(x)\n",
        "    x = tf.keras.layers.Activation('relu')(x)\n",
        "\n",
        "    if use_dropout:\n",
        "        x = tf.keras.layers.Dropout(0.5)(x)\n",
        "\n",
        "    x = ReflectionPadding2D((1,1))(x)\n",
        "    x = tf.keras.layers.Conv2D(filters=filters,\n",
        "                kernel_size=kernel_size,\n",
        "                strides=strides,)(x)\n",
        "    x = tf.keras.layers.BatchNormalization()(x)\n",
        "\n",
        "    merged = tf.keras.layers.Add()([input, x])\n",
        "    return merged"
      ],
      "metadata": {
        "id": "x0wWL4NEe3j3"
      },
      "execution_count": 12,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Generator Model\n",
        "Implemented using CNN's"
      ],
      "metadata": {
        "id": "D4BQapNwym-z"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def generatorModel():\n",
        "    \"\"\"Build generator architecture.\"\"\"\n",
        "    # Current version : ResNet block\n",
        "    inputs = tf.keras.layers.Input(shape = image_shape)    #image_shape\n",
        "\n",
        "    x = ReflectionPadding2D((3, 3))(inputs)\n",
        "    x = tf.keras.layers.Conv2D(filters=ngf, kernel_size=(7,7), padding='valid')(x)\n",
        "    x = tf.keras.layers.BatchNormalization()(x)\n",
        "    x = tf.keras.layers.Activation('relu')(x)\n",
        "\n",
        "    # Increase filter number\n",
        "    n = 2\n",
        "    for i in range(n):\n",
        "        mult = 2**i\n",
        "        x = tf.keras.layers.Conv2D(filters=ngf*mult*2, kernel_size=(3,3), strides=2, padding='same')(x)\n",
        "        x = tf.keras.layers.BatchNormalization()(x)\n",
        "        x = tf.keras.layers.Activation('relu')(x)\n",
        "\n",
        "    # Apply 9 ResNet blocks\n",
        "    mult = 2**n\n",
        "    for i in range(n_blocks_gen):\n",
        "        x = resNetBlock(x, ngf*mult, use_dropout=True)\n",
        "\n",
        "    # Decrease filter number to 3 (RGB)\n",
        "    for i in range(n):\n",
        "        mult = 2**(n - i)\n",
        "        x = tf.keras.layers.UpSampling2D()(x)\n",
        "        x = tf.keras.layers.Conv2D(filters=int(ngf * mult / 2), kernel_size=(3, 3), padding='same')(x)\n",
        "        x = tf.keras.layers.BatchNormalization()(x)\n",
        "        x = tf.keras.layers.Activation('relu')(x)\n",
        "\n",
        "    x = ReflectionPadding2D((3,3))(x)\n",
        "    x = tf.keras.layers.Conv2D(filters=output_nc, kernel_size=(7,7), padding='valid')(x)\n",
        "    x = tf.keras.layers.Activation('tanh')(x)\n",
        "\n",
        "    # Add direct connection from input to output and recenter to [-1, 1]\n",
        "    outputs = tf.keras.layers.Add()([x, inputs])\n",
        "    outputs = tf.keras.layers.Lambda(lambda z: z/2)(outputs)\n",
        "\n",
        "    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name='Generator')\n",
        "    return model "
      ],
      "metadata": {
        "id": "OfY6k0AufAhu"
      },
      "execution_count": 13,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Discriminator Model\n",
        "Implemented using CNN's"
      ],
      "metadata": {
        "id": "-ynguAA2yx7L"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def discriminatorModel():\n",
        "    \"\"\"Build discriminator architecture.\"\"\"\n",
        "    n_layers, use_sigmoid = 3, False\n",
        "    inputs = tf.keras.layers.Input(shape = input_shape_discriminator)\n",
        "\n",
        "    x = tf.keras.layers.Conv2D(filters=ndf, kernel_size=(4,4), strides=2, padding='same')(inputs)\n",
        "    x = tf.keras.layers.LeakyReLU(0.2)(x)\n",
        "\n",
        "    nf_mult, nf_mult_prev = 1, 1\n",
        "    for n in range(n_layers):\n",
        "        nf_mult_prev, nf_mult = nf_mult, min(2**n, 8)\n",
        "        x = tf.keras.layers.Conv2D(filters=ndf*nf_mult, kernel_size=(4,4), strides=2, padding='same')(x)\n",
        "        x = tf.keras.layers.BatchNormalization()(x)\n",
        "        x = tf.keras.layers.LeakyReLU(0.2)(x)\n",
        "\n",
        "    nf_mult_prev, nf_mult = nf_mult, min(2**n_layers, 8)\n",
        "    x = tf.keras.layers.Conv2D(filters=ndf*nf_mult, kernel_size=(4,4), strides=1, padding='same')(x)\n",
        "    x = tf.keras.layers.BatchNormalization()(x)\n",
        "    x = tf.keras.layers.LeakyReLU(0.2)(x)\n",
        "\n",
        "    x = tf.keras.layers.Conv2D(filters=1, kernel_size=(4,4), strides=1, padding='same')(x)\n",
        "    if use_sigmoid:\n",
        "        x = tf.keras.layers.Activation('sigmoid')(x)\n",
        "\n",
        "    x = tf.keras.layers.Flatten()(x)\n",
        "    x = tf.keras.layers.Dense(1024, activation='tanh')(x)\n",
        "    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)\n",
        "\n",
        "    model = tf.keras.models.Model(inputs=inputs, outputs=x, name='Discriminator')\n",
        "    return model"
      ],
      "metadata": {
        "id": "-5jzXQ8BfDer"
      },
      "execution_count": 14,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Function(s) to compile both independent models (only used one of em)\n",
        "The goal of a GAN is to train both the generator and discriminator simultaneously to teach my program to be able to distinguish and create false images (which are represented as sharpened up compared to the original). "
      ],
      "metadata": {
        "id": "wcatL0BRRjA-"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def generator_containing_discriminator_multiple_outputs(generator, discriminator):\n",
        "    inputs = tf.keras.layers.Input(shape=image_shape)\n",
        "    generated_images = generator(inputs)\n",
        "    outputs = discriminator(generated_images)\n",
        "    model = tf.keras.models.Model(inputs=inputs, outputs=[generated_images, outputs])\n",
        "    return model\n",
        "\n",
        "def generator_containing_discriminator(generator, discriminator):\n",
        "    inputs = tf.keras.layers.Input(shape=image_shape)\n",
        "    generated_image = generator(inputs)\n",
        "    outputs = discriminator(generated_image)\n",
        "    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)\n",
        "    return model "
      ],
      "metadata": {
        "id": "TJGhr2WERiIc"
      },
      "execution_count": 15,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Defining the Loss Functions\n",
        "Both discriminator and loss functions use binarycrossentropy loss function"
      ],
      "metadata": {
        "id": "Lt29Fe02NKVG"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)\n",
        "\n",
        "def dLoss(realPic, fakePic):\n",
        "    realLoss = cross_entropy(tf.ones_like(realPic), realPic)\n",
        "    fakeLoss = cross_entropy(tf.zeros_like(fakePic), fakePic)\n",
        "    return realLoss + fakeLoss\n",
        "\n",
        "def gLoss(fakePic):\n",
        "    return cross_entropy(tf.ones_like(fakePic), fakePic)"
      ],
      "metadata": {
        "id": "Sm2Q3bwNfGVd"
      },
      "execution_count": 16,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import files\n",
        "\n",
        "#os.chdir(dir)\n",
        "#plt.savefig('results{}.png'.format(i))\n",
        "#files.download('results{}.png'.format(i)) \n",
        "\n",
        "def generateAndSave(model, epoch, test_input):\n",
        "  # Notice `training` is set to False.\n",
        "  # This is so all layers run in inference mode (batchnorm).\n",
        "    predictions = model(test_input, training=False)\n",
        "    fp = r\"/gdrive/MyDrive/FilePath\"\n",
        "    #fig = plt.figure(figsize=(4, 4))\n",
        "\n",
        "    for i in range(predictions.shape[0]):     \n",
        "        os.chdir(fp)\n",
        "        plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))\n",
        "        files.download('image_at_epoch_{:04d}.png'.format(epoch))\n",
        "    \n",
        "    #plt.show()"
      ],
      "metadata": {
        "id": "s-Q2ibyexwus"
      },
      "execution_count": 17,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Basic Functions for image processing/checking"
      ],
      "metadata": {
        "id": "f2kkIhw-dkoZ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#found all these functions on tensorflow gan website (link provided below)\n",
        "def deprocess_image(img):\n",
        "    img = img * 127.5 + 127.5\n",
        "    return img.astype('uint8')\n",
        "\n",
        "def preprocess_image(cv_img):\n",
        "    cv_img = cv_img.resize((256,256))\n",
        "    img = np.array(cv_img)\n",
        "    img = (img - 127.5) / 127.5\n",
        "    return img\n",
        "\n",
        "def is_an_image_file(filename):\n",
        "    IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']\n",
        "    for ext in IMAGE_EXTENSIONS:\n",
        "        if ext in filename:\n",
        "            return True\n",
        "    return False\n",
        "  \n",
        "def list_image_files(directory):\n",
        "    files = sorted(os.listdir(directory))\n",
        "    return [os.path.join(directory, f) for f in files if is_an_image_file(f)]\n",
        "  \n",
        "def load_image(path):\n",
        "    img = Image.open(path)\n",
        "    return img\n",
        "def load_images(path, n_images):\n",
        "    if n_images < 0:\n",
        "        n_images = float(\"inf\")\n",
        "    A_paths, B_paths = os.path.join(path, 'A'), os.path.join(path, 'B')\n",
        "    all_A_paths, all_B_paths = list_image_files(A_paths), list_image_files(B_paths)\n",
        "    images_A, images_B = [], []\n",
        "    images_A_paths, images_B_paths = [], []\n",
        "    for path_A, path_B in zip(all_A_paths, all_B_paths):\n",
        "        img_A, img_B = load_image(path_A), load_image(path_B)\n",
        "        images_A.append(preprocess_image(img_A))\n",
        "        images_B.append(preprocess_image(img_B))\n",
        "        images_A_paths.append(path_A)\n",
        "        images_B_paths.append(path_B)\n",
        "        if len(images_A) > n_images - 1: break\n",
        "\n",
        "    return {\n",
        "        'A': np.array(images_A),\n",
        "        'A_paths': np.array(images_A_paths),\n",
        "        'B': np.array(images_B),\n",
        "        'B_paths': np.array(images_B_paths)\n",
        "    }"
      ],
      "metadata": {
        "id": "FLt-pZjYCkfP"
      },
      "execution_count": 18,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Training"
      ],
      "metadata": {
        "id": "zHmvrS7Ud45X"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import files\n",
        "fp = r\"/gdrive/MyDrive/FilePath\"\n",
        "#gonna call A = blur, B = sharp images \n",
        "\n",
        "def train(numImages, batch_size, epochs, critic_updates=5):\n",
        "      #load in photos\n",
        "      dir = TRAIN_DIR2\n",
        "      pics = load_images(dir, numImages)\n",
        "      train_x, train_y = pics['A'], pics['B'] \n",
        "\n",
        "      #initialize models for training\n",
        "      g = generatorModel()\n",
        "      d = discriminatorModel()\n",
        "      dg = generator_containing_discriminator_multiple_outputs(g, d)\n",
        "\n",
        "      optimizerD = tf.keras.optimizers.Adam(learning_rate=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)\n",
        "      optimizerDG = tf.keras.optimizers.Adam(learning_rate=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)\n",
        "\n",
        "      #training model but I am preventing disc from training since the goal is for the generator to be able to create a false image\n",
        "      d.trainable = True\n",
        "      d.compile(optimizer=optimizerD, loss=dLoss)\n",
        "      d.trainable = False\n",
        "      loss = [gLoss, dLoss]\n",
        "      lossWeights = [100, 1]\n",
        "      dg.compile(optimizer=optimizerDG, loss=loss, loss_weights=lossWeights)\n",
        "      d.trainable = True\n",
        "\n",
        "      outputTrue, outputFalse = np.ones((batch_size, 1)), - np.ones((batch_size, 1))\n",
        "      \n",
        "      for i in range(epochs):\n",
        "        permutated_indexes = np.random.permutation(train_x.shape[0])\n",
        "        lossD = []\n",
        "        dgLosses = []\n",
        "        \n",
        "        for j in range(batch_size):\n",
        "          #Prepare the batches for creating fake images...\n",
        "          batch_indexes = permutated_indexes[j * batch_size:(j + 1) * batch_size] \n",
        "          blur = train_x[batch_indexes]\n",
        "          image_full_batch = train_y[batch_indexes]\n",
        "        \n",
        "\n",
        "          #Create fake pics...\n",
        "          imagesCreated = g.predict(x=blur, batch_size=batch_size)\n",
        "    \n",
        "          #Now we'll train discriminator to assess the fake vs real pics...\n",
        "\n",
        "          for _ in range(critic_updates):\n",
        "            d_loss_real = d.train_on_batch(image_full_batch, outputTrue)\n",
        "            d_loss_fake = d.train_on_batch(imagesCreated, outputFalse) \n",
        "            d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)\n",
        "            lossD.append(d_loss)\n",
        "\n",
        "          d.trainable = False\n",
        "          #Now only train disc on fake pics and evaluate output...\n",
        "          d_on_g_loss = dg.train_on_batch(blur, [image_full_batch, outputTrue])\n",
        "          dgLosses.append(d_on_g_loss)\n",
        "\n",
        "          d.trainable = True\n",
        "          \n",
        "        #generateAndSave()\n",
        "        print(np.mean(lossD), np.mean(dgLosses))"
      ],
      "metadata": {
        "id": "NqnNuIpLCK-i"
      },
      "execution_count": 19,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def trainnn():\n",
        "  print(\"Starting Training...\")\n",
        "  train(261, BATCH_SIZE, 50, critic_updates=5) \n",
        "\n",
        "#runtime took: 5hrs and 21 min (last time i ran it but this was on google colab)"
      ],
      "metadata": {
        "id": "7nShXbWasVJX"
      },
      "execution_count": 20,
      "outputs": []
    }
  ]
}