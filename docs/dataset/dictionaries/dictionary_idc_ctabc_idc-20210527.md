# IDC: SCT Comparison Abnormalities: Data Dictionary

 

## Table of Contents

 

| Section Number | Section Title                           |
| -------------- | --------------------------------------- |
| 1              | Study                                   |
| 2              | Abnormalities from SCT comparison  read |

 



 

 

 

 



 

 



# Document Summary

 

| Property          | Value                                               |
| ----------------- | --------------------------------------------------- |
| Document Title    | IDC: SCT Comparison Abnormalities:  Data Dictionary |
| Date Created      | 05/28/2021                                          |
| Sections          | 2                                                   |
| Entries           | 10                                                  |
| Document Filename | dictionary_idc_ctabc_idc-20210527.rtf               |

 

 

 

 

 

 



 

 



# IDC: SCT Comparison Abnormalities: Data Dictionary

 

## Section 1: Study

 

| Variable            | Label                   | Description                                                  | Format Text |
| ------------------- | ----------------------- | ------------------------------------------------------------ | ----------- |
| **dataset_version** | Date Stamp for Datasets |                                                              | Char, 23    |
| **pid**             | Participant Identifier  | A unique identifier given to each  participant. For LSS participants, pid  has a format of 1xx,xxx, while for ACRIN participants, pid has a format of  2xx,xxx. | Numeric     |

 

 

 

 

 

 



 

 



## Section 2: Abnormalities from SCT comparison read

 

| Variable            | Label                                                        | Description                                                  | Format Text                                                  |
| ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **sct_ab_attn**     | Interval suspicious change in  attenuation                   | Did the abnormality have an interval  suspicious change in attenuation, for non-calcified nodules or masses with  >= 4 mm diameter? | .M="Missing"   .N="Not applicable"   1="No"   2="Yes"   9="Unable to determine" |
| **sct_ab_code**     | Abnormality code number                                      | The type of the abnormality. This should be equal to sct_ab_desc for the  corresponding abnormality in the Spiral CT Abnormality dataset (linked by  sct_ab_num). Note that the LSS  screening forms use a different numbering system than what is used in this  variable. | .M="Missing"   51="Non-calcified nodule or mass (opacity >= 4 mm diameter)"   52="Non-calcified micronodule(s) (opacity < 4 mm diameter)"   53="Benign lung nodule(s) (benign calcification)"   54="Atelectasis, segmental or greater"   55="Pleural thickening or effusion"   56="Non-calcified hilar/mediastinal adenopathy or mass (>= 10 mm on  short axis)"   57="Chest wall abnormality (bone destruction, metastasis, etc.)"   58="Consolidation"   59="Emphysema"   60="Significant cardiovascular abnormality"   61="Reticular/reticulonodular opacities, honeycombing, fibrosis,  scar"   62="6 or more nodules, not suspicious for cancer (opacity >= 4  mm)"   63="Other potentially significant abnormality above the diaphragm"   64="Other potentially significant abnormality below the diaphragm"   65="Other minor abnormality noted" |
| **sct_ab_gwth**     | Interval growth of abnormality (for  code 51 only)           | Did the abnormality have interval  growth, for non-calcified nodules or masses with >= 4 mm diameter? | .N="Not applicable"   1="No"   2="Yes"   9="Unable to determine" |
| **sct_ab_invg**     | Interval change warrants further  investigation (for oth. sig. abn. only) | Does interval change in the  abnormality warrant further investigation, for significant abnomalities other  than non-calcified nodules or masses with >= 4 mm diameter? | .M="Missing"   .N="Not applicable"   1="No"   2="Yes"   9="Unable to determine" |
| **sct_ab_num**      | Abnormality number (unique  identifier)                      | A number assigned to each  abnormality. This starts at 1 for each  participant for each study year, and counts up for each additional  abnormality that participant has in that study year. Along with pid and study_yr, this can be  used to match records in this dataset to records in the main Spiral CT  Abnormality dataset. | Numeric                                                      |
| **sct_ab_preexist** | Was abnormality pre-existing?                                | Was the abnormality pre-existing?                            | 1="No"   2="Yes"   9="Unable to determine"                   |
| **study_yr**        | Study Year of Screen                                         |                                                              | 0="T0"   1="T1"   2="T2"                                     |
| **visible_days**    | Days from randomization date to  earliest date visible       | The days from randomization until the  earliest date this abnormality was visible.   This is completed only for pre-existing abnormalities. | Numeric   .M="Missing"   .N="Not applicable"                 |

 

 

 

 

 