# IDC: Participant: Data Dictionary

 

## Table of Contents

 

| Section Number | Section Title |
| -------------- | ------------- |
| 1              | Study         |
| 2              | Demographic   |
| 3              | Smoking       |
| 4              | Screening     |
| 6              | Lung Cancer   |

 

 

 

 

 

 



 

 



# Document Summary

 

| Property          | Value                                |
| ----------------- | ------------------------------------ |
| Document Title    | IDC: Participant: Data Dictionary    |
| Date Created      | 05/28/2021                           |
| Sections          | 5                                    |
| Entries           | 33                                   |
| Document Filename | dictionary_idc_prsn_idc-20210527.rtf |

 

 

 

 

 

 



 

 



# IDC: Participant: Data Dictionary

 

## Section 1: Study

 

| Variable            | Label                   | Description                                                  | Format Text |
| ------------------- | ----------------------- | ------------------------------------------------------------ | ----------- |
| **dataset_version** | Date Stamp for Datasets |                                                              | Char, 23    |
| **pid**             | Participant Identifier  | A unique identifier given to each  participant. For LSS participants, pid  has a format of 1xx,xxx, while for ACRIN participants, pid has a format of  2xx,xxx. | Numeric     |

 

 

 

 

 

 



 

 



## Section 2: Demographic

 

| Variable   | Label                                          | Description | Format Text                                                  |
| ---------- | ---------------------------------------------- | ----------- | ------------------------------------------------------------ |
| **age**    | Age at randomization (in years; whole  number) |             | Numeric                                                      |
| **gender** | Gender                                         |             | 1="Male"   2="Female"                                        |
| **race**   | Race                                           |             | 1="White"   2="Black or African-American"   3="Asian"   4="American Indian or Alaskan Native"   5="Native Hawaiian or Other Pacific Islander"   6="More than one race"   7="Participant refused to answer"   95="Missing data form - form is not expected to ever be completed"   96="Missing - no response"   98="Missing - form was submitted and the answer was left blank"   99="Unknown/ decline to answer" |

 

 

 

 

 

 



 

 



## Section 3: Smoking

 

| Variable    | Label                | Description                                                  | Format Text              |
| ----------- | -------------------- | ------------------------------------------------------------ | ------------------------ |
| **cigsmok** | Smoking status at T0 | Cigarette smoking status (current vs  former) at randomization. Former  smokers must have quit within 15 years of eligibility determination to have  been eligible for the trial. | 0="Former"   1="Current" |

 

 

 

 

 

 



 

 



## Section 4: Screening

 

| Variable        | Label                                                        | Description                                                  | Format Text                                                  |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **scr_days0-2** | Days since randomization at time of  screening (for T0, T1, and T2 exams) | Days from randomization to the date  of the screening exam.      Contains information on either the spiral CT or the chest x-ray screening  exam, depending on which study arm the participant was randomized to (see  rndgroup). | Numeric   .E="Screen date after lung cancer diagnosis"   .N="No screen date on record"   .W="Wrong Screen Administered" |
| **scr_iso0-2**  | Result of isolation screen (for T0,  T1, and T2 exams)       | Isolation read of the image(s) from  the screening exam before comparing  with prior screening images or other available prior images.      Contains information on either the spiral CT or the chest x-ray screening  exam, depending on which study arm the participant was randomized to (see  rndgroup). | 1="Negative screen, no  significant abnormalities"   2="Negative screen, minor abnormalities not suspicious for lung  cancer"   3="Negative screen, significant abnormalities not suspicious for lung  cancer"   4="Positive, Change Unspecified, nodule(s) >= 4 mm or enlarging  nodule(s), mass(es), other non-specific abnormalities suspicious for lung  cancer"   10="Inadequate Image"   11="Not Compliant - Left Study"   13="Not Expected - Cancer before screening window"   14="Not Expected - Death before screening window"   15="Not Compliant - Refused a screen"   17="Not Compliant - Wrong Screen"   23="Not Expected - Cancer in screening window"   24="Not Expected - Death in screening window"   95="Not Compliant - Erroneous Report of Lung Cancer Before Screen (LSS  Only)"   97="Not Compliant - Form Not Submitted, Window Closed" |
| **scr_res0-2**  | Results of screening (for T0, T1, and  T2 exams)             | Official result of the screening  exam, after comparing the current image(s) with prior screening images or  other available prior images.       Participants with positive screens (scr_resX = 4, 5, or 6) were advised to  receive a diagnostic evaluation for lung cancer with their personal  physician.      Contains information on either the spiral CT or the chest x-ray screening  exam, depending on which study arm the participant was randomized to (see  rndgroup). | 1="Negative screen, no  significant abnormalities"   2="Negative screen, minor abnormalities not suspicious for lung  cancer"   3="Negative screen, significant abnormalities not suspicious for lung  cancer"   4="Positive, Change Unspecified, nodule(s) >= 4 mm or enlarging  nodule(s), mass(es), other non-specific abnormalities suspicious for lung  cancer"   5="Positive, No Significant Change, stable abnormalities potentially  related to lung cancer, no significant change since prior screening  exam."   6="Positive, other"   10="Inadequate Image"   11="Not Compliant - Left Study"   13="Not Expected - Cancer before screening window"   14="Not Expected - Death before screening window"   15="Not Compliant - Refused a screen"   17="Not Compliant - Wrong Screen"   23="Not Expected - Cancer in screening window"   24="Not Expected - Death in screening window"   95="Not Compliant - Erroneous Report of Lung Cancer Before Screen (LSS  Only)"   97="Not Compliant - Form Not Submitted, Window Closed" |

 

 

 

 

 

 



 

 



## Section 6: Lung Cancer

 

| Variable           | Label                                                        | Description                                                  | Format Text                                                  |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **can_scr**        | Result of screen associated with the  first confirmed lung cancer diagnosis | Indicates whether the cancer followed  a positive, negative, or missed screen, or whether it occurred after the  screening years. | 0="No Cancer"   1="Positive Screen"   2="Negative Screen"   3="Missed Screen"   4="Post Screening" |
| **canc_free_days** | Days from randomization to date when  participant was last known to be free from lung cancer | Days until the date the participant  was last known to be free of lung cancer.   This date comes from a participant study update form, except for ACRIN  participants with cancer, where the cancer diagnosis date is used.      Do not use this variable to calculate follow-up time for lung cancer  incidence rates, Cox regressions, etc.   Instead, use fup_days for non-cases and candx_days for cases. | Numeric                                                      |
| **canc_rpt_link**  | Is the diagnosis of lung cancer  associated with a positive screen? | Is the diagnosis of lung cancer  associated with a positive screen? A  positive screen and a cancer diagnosis are associated based on a linking  algorithm using diagnostic procedures. | 0="No"   1="Yes"                                             |
| **cancyr**         | Study year associated with first  confirmed lung cancer      | Study year associated with the first  confirmed lung cancer. A cancer  associated with a positive screen is  assigned the study year of that screen. | .N="Not Applicable"   0="T0"   1="T1"   2="T2"   3="T3"   4="T4"   5="T5"   6="T6"   7="T7" |
| **candx_days**     | Days from randomization to first  diagnosis of lung cancer   |                                                              | Numeric   .N="No diagnosis date on record"                   |
| **de_grade**       | Lung cancer grade                                            |                                                              | .N="Not Applicable"   1="Grade Cannot Be Assessed (GX) "   2="Well Differentiated (G1)"   3="Moderately Differentiated (G2)"   4="Poorly Differentiated (G3)"   5="Undifferentiated (G4)"   6="Unspecified in Pathology Report"   8="Unknown"   9="Missing" |
| **de_stag**        | Lung cancer Stage                                            | Lung cancer stage (AJCC 6), combining  clinical and pathologic staging information. | .M="Missing"   .N="Not Applicable"   110="Stage IA"   120="Stage IB"   210="Stage IIA"   220="Stage IIB"   310="Stage IIIA"   320="Stage IIIB"   400="Stage IV"   888="TNM not available"   900="Occult Carcinoma"   994="Carcinoid, cannot be assessed"   999="Unknown, cannot be assessed" |
| **de_stag_7thed**  | AJCC 7th edition stage                                       | Stage of first primary lung cancer,  based on AJCC 7th edition stage | .M="Missing"   .N="Not Applicable"   110="Stage IA"   120="Stage IB"   210="Stage IIA"   220="Stage IIB"   310="Stage IIIA"   320="Stage IIIB"   400="Stage IV"   900="Occult Carcinoma"   999="Unknown, cannot be assessed" |
| **de_type**        | Lung cancer type from ICD-O-3  morphology                    | Lung cancer type from ICD-O-3  morphology. For LSS participants, this  is recorded separately from the complete ICD code, and represents the best  information available on the type of the cancer. | Numeric   .M="Missing"   .N="Not Applicable"                 |
| **lesionsize**     | Pathology lesion size of tumor in  millimeters               |                                                              | Numeric   .M="Missing"   .N="Not Applicable"                 |
| **loccar**         | Cancer in Carina                                             | Was the primary tumor located in the  carina? A tumor may be located in more  than one location. | .N="Not Applicable"   0="No"   1="Yes"                       |
| **loclhil**        | Cancer in Left Hilum                                         | Was the primary tumor located in the  left hilum? A tumor may be located in  more than one location. | .N="Not Applicable"   0="No"   1="Yes"                       |
| **loclin**         | Cancer in Lingula                                            | Was the primary tumor located in the  lingula? A tumor may be located in  more than one location. | .N="Not Applicable"   0="No"   1="Yes"                       |
| **locllow**        | Cancer in Left lower lobe                                    | Was the primary tumor located in the  left lower lobe? A tumor may be  located in more than one location. | .N="Not Applicable"   0="No"   1="Yes"                       |
| **loclmsb**        | Cancer in Left main stem bronchus                            | Was the primary tumor located in the  left main stem bronchus? A tumor may  be located in more than one location. | .N="Not Applicable"   0="No"   1="Yes"                       |
| **loclup**         | Cancer in Left upper lobe                                    | Was the primary tumor located in the  left upper lobe? A tumor may be  located in more than one location. | .N="Not Applicable"   0="No"   1="Yes"                       |
| **locmed**         | Cancer in Mediastinum                                        | Was the primary tumor located in the  mediastinum? A tumor may be located in  more than one location. | .N="Not Applicable"   0="No"   1="Yes"                       |
| **locoth**         | Cancer in Other Location                                     | Was the primary tumor located in an  other specified location? A tumor may  be located in more than one location. | .N="Not Applicable"   0="No"   1="Yes"                       |
| **locrhil**        | Cancer in Right Hilum                                        | Was the primary tumor located in the  right hilum? A tumor may be located in  more than one location. | .N="Not Applicable"   0="No"   1="Yes"                       |
| **locrlow**        | Cancer in Right lower lobe                                   | Was the primary tumor located in the  right lower lobe? A tumor may be  located in more than one location. | .N="Not Applicable"   0="No"   1="Yes"                       |
| **locrmid**        | Cancer in Right middle lobe                                  | Was the primary tumor located in the  right middle lobe? A tumor may be  located in more than one location. | .N="Not Applicable"   0="No"   1="Yes"                       |
| **locrmsb**        | Cancer in Right main stem bronchus                           | Was the primary tumor located in the  right main stem bronchus? A tumor may  be located in more than one location. | .N="Not Applicable"   0="No"   1="Yes"                       |
| **locrup**         | Cancer in Right upper lobe                                   | Was the primary tumor located in the  right upper lobe? A tumor may be  located in more than one location. | .N="Not Applicable"   0="No"   1="Yes"                       |
| **locunk**         | Cancer in Unknown location                                   | Was the primary tumor located in an  unknown location? A tumor may be  located in more than one location. | .N="Not Applicable"   0="No"   1="Yes"                       |

 

 

 

 

 