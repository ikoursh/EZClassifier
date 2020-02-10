# EZClassifier
A simple and easy TensorFlow classifier

## How To Install 
For total beginners, it is recommended to go through everything, but for people with experience in IntelliJ and git - skip this.
### How to Setup The Project From Github
1. Download and install [InteliJ](https://www.jetbrains.com/idea/download/#section=windows) or any other Java IDE that supports Maven.
**IF YOU ARE A STUDENT GO [HERE](https://www.jetbrains.com/shop/eform/students) TO APPLY FOR A FREE JetBrains PROFESSIONAL LICENCE**.
2. Download and install [GIT](https://git-scm.com/downloads).
3. In the IntelliJ homepage select "checkout from version control", "git". In the URL section enter: https://github.com/ikoursh/EZClassifier/ and follow the instructions to cloning a project.
4. You are done! now you can skip to the section that interests you.
### How to Setup Python
Download the latest python version [here](https://www.python.org/).

## Setup A Teachable Machine Model
**Disclaimer:**  
Only applies to models trained via [this](https://teachablemachine.withgoogle.com/train/image) demo. For other models trained by the teachable machine go to the general teachable machine instructions.  
  
With that out of the way let's get started.  
1. Go to https://teachablemachine.withgoogle.com/train/image and train a model.
2. Select "Export Model", "TensorFlow", ensure that "Keras" is selected, and finally press "Download My Model".  
3. Extract the zip file. in it, there should be 2 files: labels.txt and keras_model.h5.  
4. Open the IntelliJ project you cloned from the "How to Setup The Project From Github" section.  
5. Right-click on the "model-converter" folder on the left panel and select "show in explorer/finder/Linux alt".  
6. In the new explorer window double click on the "model-converter" folder, and then on the "model converter.exe".  
* If you see an error message "blablabla.dll not found" **INSTALL PYTHON AND TRY AGAIN**. If the error persists got to the wiki article for building the model converter yourself.  
7. in the select file prompt open the keras_model.h5 you extracted.
8. When prompted to save a file save it somewhere.
9. You are done! you can now open the example main function, replace the path to the classifier and the labels file with the ones you downloaded/created, and then run predictions on images. Good Job!
