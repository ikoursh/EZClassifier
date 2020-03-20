# EZClassifier
A simple and easy TensorFlow classifier

## How To Install 
For total beginners, it is recommended to go through everything, but for people with experience in IntelliJ and git - skip this.
### How to Setup The Project From Github
1. Download and install [InteliJ](https://www.jetbrains.com/idea/download/#section=windows) or any other Java IDE that supports Maven.
**if you are a student go [here](https://www.jetbrains.com/shop/eform/students) to apply for a free JetBrains professional lisence**.
2. Download and install [GIT](https://git-scm.com/downloads).
3. In the IntelliJ homepage select "checkout from version control", "git". In the URL section enter: https://github.com/ikoursh/EZClassifier/ and follow the instructions to cloning a project.
4. You are done! now you can skip to the section that interests you.
### How to Setup Python
Download the latest python version [here](https://www.python.org/).

## Setup A Standard Teachable Machine Image Model
**Disclaimer:**  
Only applies to models trained via [this](https://teachablemachine.withgoogle.com/train/image) demo. For other image-based models trained by the teachable machine go to the general imaged based model instructions.  
  
With that out of the way let's get started.  
1. Go to https://teachablemachine.withgoogle.com/train/image and train a model.
2. Select "Export Model", "TensorFlow", ensure that "Keras" is selected, and finally press "Download My Model".  
3. Extract the zip file. in it, there should be 2 files: labels.txt and keras_model.h5.  
4. Open the IntelliJ project you cloned from the "How to Setup The Project From Github" section.  
5. Right-click on the "model-converter" folder on the left panel and select "show in explorer/finder/Linux alt" (or go to `EZClassifier/model-converter`).  
6. In the new Explorer window double click on the "model-converter" folder, and then on the "model converter.exe".  
* If you see an error message "blablabla.dll not found" **INSTALL PYTHON AND TRY AGAIN**. If the error persists got to the wiki article for building the model converter yourself.  
7. in the select file prompt open the keras_model.h5 you extracted.
8. When prompted to save a file save it somewhere.
9. You are done! you can now open the example main function, replace the path to the classifier and the labels file with the ones you downloaded/created, and then run predictions on images. Good Job!

## Setup Any Image-Based Tensorflow Model
Running a model created by the teachable machine is nice, but if you want other models or a different teachable machine example follow this guide.
1. Like in the more basic example we need an h5 model. If you are using the teachable machine follow instructions 2-8 but with a teachable machine model of your choice, then jump to step 2. If you want to use your own TensorFlow model I will now instruct you how to get the required model.h5 and labels.txt files:
    1. Run `model.save('model.h5')` on your TensorFlow model to create the model.h5 file.
    2. For the labels.txt, create a standard text file so that each line represents the name of an output class. For example, if I have 2 output nodes: the first representing the probability that the picture contains "Bob", and the second the probability that it contains "Malinda", an example file could look like:
      ```
      Bob
      Malinda
      ```
2. Now you need to get the model info for your model. Open cmd and move to the directory that get-model-config.exe is in, usually it should be `EZClassifier\get-model-config`. Now execute: `get_model_config.exe`, select the converted PB model file. The program should (when done) output "input operation" and "output operation".
3. Good Job! you can now create an instance of EZClasifier with a custom model. Note: you are expected to know the image dimensions your model expects if you don't - try 224,224
