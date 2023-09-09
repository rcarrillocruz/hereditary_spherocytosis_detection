
# Hereditary Spherocytosis - HS Detect Web App
AI-Powered medical tool for early detection of Hereditary Spherocytosis and other conditions causing similar symptoms.

# Table of Contents

1. [What is Hereditary Spherocytosis (HS)?](#id-section1)
2. [What does HS Detect Web App do?](#id-section2)
3. [Behind the Scenes: A Deep Learning approach for Segmentation of Red Blood Cell Images and HS Detection](#id-section3)
4. [Getting Started](#id-section4)
5. [Streamlit Cloud](#id-section5)
6. [Help us improve the project](#id-section6)
7. [Authors](#id-section7)
8. [Version history](#id-section8)
9. [License](#id-section9)
10. [Acknowledgments](#id-section10)


<div id='id-section1'/>

## What is Hereditary Spherocytosis (HS)? 💉 🩸

According to [Nemours KidsHealth](https://kidshealth.org/en/parents/hereditary-spherocytosis.html), Hereditary Spherocytosis (HS) is an inherited blood disorder. It happens because of a problem with the red blood cells (RBCs). Instead of being shaped like a disk, the cells are round like a sphere.

These red blood cells (called **spherocytes**) are more fragile than disk-shaped RBCs. They break down faster and more easily than normal RBCs. This breakdown leads to [anemia](https://kidshealth.org/en/parents/anemia.html) (not enough RBCs in the body) and other medical problems. Anemia caused by breaking down of RBCs is called **[hemolytic anemia](https://kidshealth.org/en/parents/anemia-hemolytic.html)**.

Symptoms may range from mild to severe. Treatments can help with symptoms.

![](https://kidshealth.org/content/dam/patientinstructions/en/images/spherocytosis_a_enIL.jpg)

<div id='id-section2'/>

## What does HS Detect Web App do? 🚀🚀

[Wikidoc](https://www.wikidoc.org/index.php/Hereditary_spherocytosis_laboratory_findings) indicates that "The initial laboratory testing for hereditary spherocytosis include; [complete blood count (CBC)](https://www.wikidoc.org/index.php/Complete_blood_count "Complete blood count"), [mean corpuscular hemoglobin concentration (MCHC)](https://www.wikidoc.org/index.php/Mean_corpuscular_hemoglobin_concentration "Mean corpuscular hemoglobin concentration"), [blood smear review](https://www.wikidoc.org/index.php/Blood_film "Blood film"), [hemolysis](https://www.wikidoc.org/index.php/Hemolysis "Hemolysis") [testing](https://www.wikidoc.org/index.php/Test "Test") and [coombs testing](https://www.wikidoc.org/index.php/Coombs_test "Coombs test")".

A blood smear is a slide made from a drop of blood, that allows the cells to be examined microscopically, done to investigate hematological problems (disorders of the blood itself) and, occasionally, to look for parasites within the blood.

While usually this review is done by pathologists (by identifying different characteristics of RBCs as shown on the image above), this is a time-consuming procedure, dependent on his skills.

**HS Detect Web App is a tool developed for early detection of HS online** : you only need to upload a blood smear image of a patient on our web app and, through the use of deep learning (DL) techniques and convolutional neural networks (CNN), you will receive information regarding the chances the patient may or may not have hereditary spherocytosis by checking the presence of spherocytes.

Both patients and physicians can use HS Detect!

**DISCLAIMER: HS Detect Web App is not meant to replace diagnosis or confirmation tests of HS. Please consult a medical professional / doctor for further testing and treatment.**

<div id='id-section3'/>

## Behind the Scenes: A Deep Learning approach for Segmentation of Red Blood Cell Images and HS Detection 👨‍🔬👩‍🔬

### Data :

* [Dataset A: 186 digital images of MGG-stained blood smears from five patients with hereditary spherocytosis](https://data.mendeley.com/datasets/c37wnbbd3c/1) : This dataset contains 186 digital images of May Grünwald-Giemsa (MGG)-stained peripheral blood (PB) smears. They were compiled during the daily work in the Core Laboratory at the Hospital Clínic of Barcelona from five patients with the diagnosis of hereditary spherocytosis. The digital images were acquired using a microscope with 1,000x magnification (Olympus BX43) and a digital camera (Olympus DP73). The images in the dataset are of format JPG (RGB, 2,400 x 1,800 pixels).
* [Chula RBC-12-Dataset](https://github.com/Chula-PIC-Lab/Chula-RBC-12-Dataset) : This is a dataset of red blood cell (RBC) blood smear images used in "Red Blood Cell Segmentation with Overlapping Cell Separation and Classification from an Imbalanced Dataset", containing 12 classes of RBC types consisting of 706 smear images that contain over 20,000 RBC cells. The dataset was collected at the Oxidation in Red Cell Disorders Research Unit, Chulalongkorn University in 2019 using a DS-Fi2-L3 Nikon microscope at 1000x magnification.

### Bibliography :

* Delgado-Ortet M, Molina A, Alférez S, Rodellar J, Merino A. A Deep Learning Approach for Segmentation of Red Blood Cell Images and Malaria Detection. Entropy (Basel). 2020 Jun 13;22(6):657. doi: 10.3390/e22060657. PMID: 33286429; PMCID: PMC7517192. ([link](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7517192/))
* Korranat Naruenatthanaset, Thanarat H. Chalidabhongse, Duangdao Palasuwan, Nantheera Anantrasirichai, Attakorn Palasuwan. Red Blood Cell Segmentation with Overlapping Cell Separation and Classification on Imbalanced Dataset. 2023. [arXiv:2012.01321](https://arxiv.org/abs/2012.01321). [https://doi.org/10.48550/arXiv.2012.01321](https://doi.org/10.48550/arXiv.2012.01321). ([link](https://arxiv.org/abs/2012.01321))
* **TBC**

### The Model :

**The Convolutional Neural Network (CNN) architecture behind HS Detect has been developed for Image Segmentation**, in order to detect RBCs and classify them (assigning a class label to each segment).

One of the main inspirations for the model is the work of Maria Delgado-Ortet, Angel Molina, Santiago Alférez, José Rodellar, and Anna Merino (2020), who developed a three-stage pipeline to (1) segment erythrocytes, (2) crop and mask them, and (3) classify them into malaria infected or not.

One of the datasets used during the training (Dataset A) was in fact also collected by this research team at the Hospital Clínic of Barcelona.

<div id='id-section4'/>

## Getting Started  🖥️

### Dependencies
```
tensorflow
scikit-learn
numpy
pandas
matplotlib
seaborn
scikit-learn
```
### Installing

* How/where to download your program
* Any modifications needed to be made to files/folders
* Install all streamlit requirements by run the following command

```
pip install requirements.txt
```

### Executing program

* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

### Run Streamlit

To access and use the application, donwoload or clone the repository and then run the command below.
```
streamlit run app.py
```
Finally browse the link provided in your browser.


<div id='id-section5'/>

## Streamlit Cloud 💻

The application wiil be deployed in Streamlit Cloud.
You can access here: *not available right now*

<div id='id-section6'/>

## Help us improve the project 🔌

### Issues
Incase you have any difficulties or issues while trying to run the app you can raise it on the issues section.

### Pull Requests

If you have something to add or new idea to implement, you are welcome to create a pull requests on improvement.

### Give it a Star

If you find this repo useful , give it a star so as many people can get to know it.

<div id='id-section7'/>

## Authors 👨‍💻👩‍💻

This Web App is going to be launched in October 2023 as a final project for [Le Wagon](https://www.lewagon.com/)'s Data Science Bootcamp (Batch #1275).

Project Leader : [Ricardo Carrillo Cruz](https://github.com/rcarrillocruz)

Team Members :
 - [Claudia Beltrán Bocanegra](https://github.com/Clausen1990)
 - [Alvaro Carranza](https://github.com/Alvaro2c)
 - [Mourad El Moufid](https://github.com/MouradElMoufid)
 - [Afonso Pinheiro](https://github.com/afonsorpinheiro)

<div id='id-section8'/>

## Version History 📑

* 0.2
    * *TBC*
    * See [commit change]() or See [release history]()
* 0.1
    * *TBC*

<div id='id-section9'/>

## License 📖

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

<div id='id-section10'/>

## Acknowledgments 🏅🏅

Inspiration, code snippets, etc.
* [TBC](TBC)
