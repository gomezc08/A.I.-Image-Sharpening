This project was completed during my 2022 summer internship at HP. I began developing and implementing an A.I. image sharpening tool using generative adversarial networks (GAN). Below is an example of my tool in action on one of my personal images...


![Picture11](https://user-images.githubusercontent.com/97072661/183763752-26838eb6-bf97-4869-bfa1-345a80e56d77.jpg)
![Picture22](https://user-images.githubusercontent.com/97072661/183763140-0fee33f0-12d2-4a16-a354-ba0f3c6d51d2.jpg)

I used GoogleColab as my source-code editor since it has a built in GPU. If you try to run this on the free version, it will probably produce an error since my models utilized a lot of convolutional layers and need a lot of GPU memory.

My project utilized 2 main models: Discriminator and Generator. From there, I trained both of them simultaneously (in several stages) to train my model to be able to create its own false image (which represents a sharpened up image compared to the orignal).
