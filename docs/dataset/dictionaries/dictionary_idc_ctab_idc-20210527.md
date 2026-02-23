# IDC: SCT Abnormalities: Data Dictionary

 

## Table of Contents

 

| Section Number | Section Title                   |
| -------------- | ------------------------------- |
| 1              | Study                           |
| 2              | Abnormalities from CT Screening |

 

 

 

 

 

 



 

 



# Document Summary

 

| Property          | Value                                    |
| ----------------- | ---------------------------------------- |
| Document Title    | IDC: SCT Abnormalities: Data  Dictionary |
| Date Created      | 05/28/2021                               |
| Sections          | 2                                        |
| Entries           | 12                                       |
| Document Filename | dictionary_idc_ctab_idc-20210527.rtf     |

 

 

 

 

 

 



 

 



# IDC: SCT Abnormalities: Data Dictionary

 

## Section 1: Study

 

| Variable            | Label                   | Description                                                  | Format Text |
| ------------------- | ----------------------- | ------------------------------------------------------------ | ----------- |
| **dataset_version** | Date Stamp for Datasets |                                                              | Char, 23    |
| **pid**             | Participant Identifier  | A unique identifier given to each  participant. For LSS participants, pid  has a format of 1xx,xxx, while for ACRIN participants, pid has a format of  2xx,xxx. | Numeric     |

 

 

 

 

 

 



 

 



## Section 2: Abnormalities from CT Screening

 

| Variable                 | Label                                                        | Description                                                  | Format Text                                                  |
| ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **sct_ab_desc**          | Abnormality description                                      | The type of the abnormality. Note that the LSS screening forms use a  different numbering system than what is used in this variable. | 51="Non-calcified nodule or mass  (opacity >= 4 mm diameter)"   52="Non-calcified micronodule(s) (opacity < 4 mm diameter)"   53="Benign lung nodule(s) (benign calcification)"   54="Atelectasis, segmental or greater"   55="Pleural thickening or effusion"   56="Non-calcified hilar/mediastinal adenopathy or mass (>= 10 mm on  short axis)"   57="Chest wall abnormality (bone destruction, metastasis, etc.)"   58="Consolidation"   59="Emphysema"   60="Significant cardiovascular abnormality"   61="Reticular/reticulonodular opacities, honeycombing, fibrosis,  scar"   62="6 or more nodules, not suspicious for cancer (opacity >= 4  mm)"   63="Other potentially significant abnormality above the diaphragm"   64="Other potentially significant abnormality below the diaphragm"   65="Other minor abnormality noted" |
| **sct_ab_num**           | Abnormality number (unique  identifier)                      | A number assigned to each  abnormality. This starts at 1 for each  participant for each study year, and counts up for each additional  abnormality that participant has in that study year. Along with pid and study_yr, this can be  used to match abnormality records in this dataset to records in the Spiral CT  Comparison Read Abnormalities dataset . | Numeric                                                      |
| **sct_epi_loc**          | Location of epicenter                                        | Location of epicenter for  non-calcified nodules or masses with >= 4 mm diameter. | .N="Not Applicable (sct_ab_desc  is not 51)"   1="Right Upper Lobe"   2="Right Middle Lobe"   3="Right Lower Lobe"   4="Left Upper Lobe"   5="Lingula"   6="Left Lower Lobe"   8="Other (Specify in comments)" |
| **sct_found_after_comp** | Was the abnormality not identified  until the comparison with historical images? | Was the abnormality not identified  until the comparison with historical images?   Abnormalities are in this dataset regardless of whether they were  found on the initial read of the screen or during the comparison with prior  images. | .M="Missing"   0="Identified on first look"   1="Found after comparison" |
| **sct_long_dia**         | Longest diameter (in mm)                                     | Longest diameter in millimeters for  non-calcified nodules or masses with >= 4 mm diameter. | Numeric   .N="Not applicable (sct_ab_desc is not 51)"   .S="Unable to determine" |
| **sct_margins**          | Margins                                                      | Description of margins for  non-calcified nodules or masses with >= 4 mm diameter. | .N="Not applicable (sct_ab_desc  is not 51)"   1="Spiculated (Stellate)"   2="Smooth"   3="Poorly defined"   9="Unable to determine" |
| **sct_perp_dia**         | Longest perpendicular diameter (same  CT slice in mm)        | Longest diameter perpendicular to the  longest overall diameter, in millimeters, for non-calcified nodules or masses  with >= 4 mm diameter. | Numeric   .N="Not applicable (sct_ab_desc is not 51)"   .S="Unable to determine" |
| **sct_pre_att**          | Predominant attenuation                                      | Predominant attenuation for  non-calcified nodules or masses with >= 4 mm diameter. | .M="Missing"   .N="Not applicable (sct_ab_desc is not 51)"   1="Soft Tissue"   2="Ground glass"   3="Mixed"   4="Fluid/water"   6="Fat"   7="Other"   9="Unable to determine" |
| **sct_slice_num**        | CT slice number containing  abnormality's greatest diameter  | The CT slice number containing the  abnormality's greatest diameter, for non-calcified nodules or masses with  >= 4 mm diameter. | Numeric   .N="Not Applicable (sct_ab_desc is not 51)"   999="Missing" |
| **study_yr**             | Study year of screen                                         |                                                              | 0="T0"   1="T1"   2="T2"                                     |

 

 

 

 

 