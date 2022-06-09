# Mediapipe_Generate_UVS
![Intro](https://github.com/alex1779/Mediapipe_Generate_UVS/blob/master/imgs/1.jpg)
## Requirements
```
numpy==1.22.3
mediapipe==0.8.9.1
opencv-python==4.5.5.64
imutils==0.5.4
```

## Installation on Windows using Anaconda
```
conda create -n Mediapipe_Generate_UVS -y && conda activate Mediapipe_Generate_UVS && conda install python=3.9.7 -y
git clone https://github.com/alex1779/Mediapipe_Generate_UVS.git
cd Mediapipe_Generate_UVS
pip install -r requirements.txt

```
## For running
Use the parameter -i to put the correct path where your image is located, like the example below

```
python main.py -i input/01.jpg
```

![Intro](https://github.com/alex1779/Mediapipe_Generate_UVS/blob/master/imgs/2.jpg)

Press S to save the image in 'ouput' folder. if there is no folder named 'output' then you must create it next to 'input' folder.

## How Works


## Things to improve

Obviously this needs to be improved since it does not return a correct output. The image should be complete and missing 'triangles'. This is because a direct triangulation of the original image is performed. Perhaps to solve the 'empty triangles' it could be filled taking as reference the opposite side of the face.

![Intro](https://github.com/alex1779/Mediapipe_Generate_UVS/blob/master/imgs/3.jpg)







## License

Many parts taken from the cpp implementation from github.com/google/mediapipe

Copyright 2020 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.






