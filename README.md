# Image-Processing-SCC0251---Final-project
Final graduation project of the SCC0251 - Image Processing discipline.

## Students

  [Caio Basso](https://github.com/caioadb)

  [Gabriel Garcia Lorencetti](https://github.com/gabrielgarcia7)

  [Leonardo Rossi Luiz](https://github.com/leonrossi)

  [Witor Matheus Alves de Oliveira](https://github.com/witorMao)

## Main objective
The main objective of this project is to build a program able to recognize letters of a braille text present in an image digitally generated, i.e, given an input image, containing a text in braille, perform the translation to the alphabetic writing system.

## Description of input images
The images we will use will be images with braille text, with a good contrast between the background and the text, regardless of the chosen colors. Below are some examples, with black symbols and white background:

  ### ![Image 1](https://fontmeme.com/temporary/d87d6a58412750d2a0404687d22e4a1f.png)
  Figure 1 - “bruh” braille text example.
  ### ![Image 2](https://fontmeme.com/temporary/810b93ad97fffc71b12c254742b15cd4.png)
  Figure 2 - “image processing” braille text example.
  
Figures 1 and 2 were generated using the [Fonte meme](https://fontmeme.com/braille/) website.
Texts were also generated with the [Braille library](https://github.com/AaditT/braille), available on GitHub. Using it, we can obtain white symbols with a black background, still maintaining a good contrast.

  ### ![Image 3]()
  Figure 3 - “hello world” braille text example.
  
## Steps to reach the objective
First, we will apply enhancements to the images to highlight the color changes between the symbols and the background in order to obtain an image only with levels 0 and 255 of color intensity. We will use the piecewise intensity transformation function, selecting the appropriate threshold for this.

Second, we will apply an image segmentation, so we can separate the symbols within each sentence of text and then pass those symbols to a final step in the program. Therefore, as we know that a set of braille symbols can vary between them, the algorithm will be able to identify different distances - letter to letter, word to word or just distances between each symbol that make up a letter - and, with that information, be able to capture every feature of the image. 

Finally, we will apply the image description to split the input images, so that we can compare with a dictionary and recognize the letters that the symbols represent. The framework Bag of Features (or Bag of Visual Words) will be used, which, based on patterns (in our context, alphabetic letters in the braille system), learns descriptors, which will be used to identify symbols in images.

