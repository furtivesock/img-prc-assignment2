# Image processing assignment 2

Source code for second assignment in Image processing course at Polytech Paris-Saclay engineering school.

It implements Hough Transform technique applied to circles.

This file contains instructions for using it.

## Prerequisites

- Python 3

## File tree

```sh
img-prc-assignment2
├──images # Folder containing the required images
├──main.py # Exercise 2
└──main_optimized.py # Exercise 3
```

## Installation

1. Clone the project first

```sh
git clone https://github.com/furtivesock/img-prc-assignment2.git
```

2. As images are not included in the repository, you need to download the following files and put them in the project root :
  - [Images with circles to detect](http://hebergement.u-psud.fr/emi/TIPolytech/TP2/images)

You can see the organization needed [above](#file-tree).

You can also add the images using curl :
```sh
cd img-prc-assignment2
mkdir images
cd images
curl http://hebergement.u-psud.fr/emi/TIPolytech/TP2/images/coins.png --output coins.png
curl http://hebergement.u-psud.fr/emi/TIPolytech/TP2/images/coins2.jpg --output coins2.jpg
curl http://hebergement.u-psud.fr/emi/TIPolytech/TP2/images/four.png --output four.png
curl http://hebergement.u-psud.fr/emi/TIPolytech/TP2/images/fourn.png --output fourn.png
curl http://hebergement.u-psud.fr/emi/TIPolytech/TP2/images/MoonCoin.png --output MoonCoin.png
```
## Usage

### Exercise 2

1. Move to the root directory

```sh
cd img-prc-assignment2/
```

2. Run the script :

```sh
python3 main.py
```

3. If you are on a Unix-like operating system, you can also make it executable :

```
chmod +x main.py
./main.py
```

After a few seconds of processing, the final image should appear.

3. Set the optional parameters if you want :
```sh
python3 main.py [-h] [--images IMAGES [IMAGES ...]] [-s --save]
```

Or

```sh
./main.py [-h] [--images IMAGES [IMAGES ...]] [-s --save]
```

- `-s` or `--save` : Save the results in an `/output` folder
- `--images` : A list of images to process.
 If not specified, the script will use the images contained in the `images` folder, except for the big `coins2.jpg`.

### Exercise 2

1. Move to the `src/` directory

```sh
cd img-prc-assignment2/
```

2. Run the script :

```sh
python3 main_optimized.py
```

Or

```
chmod +x main_optimized.py
./main_optimized.py
```