# IDC: SCT Screening: Data Dictionary

 

## Table of Contents

 

| Section Number | Section Title       |
| -------------- | ------------------- |
| 1              | Study               |
| 2              | Spiral CT Screening |

 

 

 

 

 

 



 

 



# Document Summary

 

| Property          | Value                                  |
| ----------------- | -------------------------------------- |
| Document Title    | IDC: SCT Screening: Data Dictionary    |
| Date Created      | 05/28/2021                             |
| Sections          | 2                                      |
| Entries           | 17                                     |
| Document Filename | dictionary_idc_screen_idc-20210527.rtf |

 

 

 

 

 

 



 

 



# IDC: SCT Screening: Data Dictionary

 

## Section 1: Study

 

| Variable            | Label                   | Description                                                  | Format Text |
| ------------------- | ----------------------- | ------------------------------------------------------------ | ----------- |
| **dataset_version** | Date Stamp for Datasets |                                                              | Char, 23    |
| **pid**             | Participant Identifier  | A unique identifier given to each  participant. For LSS participants, pid  has a format of 1xx,xxx, while for ACRIN participants, pid has a format of  2xx,xxx. | Numeric     |

 

 

 

 

 

 



 

 



## Section 2: Spiral CT Screening

 

| Variable                | Label                                                        | Description                                                  | Format Text                                                  |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **ct_recon_filter1-4**  | CT reconstruction algorithm / filter                         | What CT reconstruction algorithm /  filter was used for the screen?      These variables come from the data collection forms. They may disagree with data extracted from  the CT images' DICOM headers. Header  data may be obtained from the NLST CT image collection at TCIA or from ACRIN. | .M="Missing or less than 4  algorithms/filters"   1="GE Bone"   2="GE Standard"   3="GE, other"   4="Phillips D"   5="Phillips C"   6="Phillips, other"   7="Siemens B50F"   8="Siemens B30"   9="Siemens, other"   10="Toshiba FC10"   11="Toshiba FC51"   12="Toshiba, other" |
| **ctdxqual**            | Overall diagnostic quality of CT  examination                |                                                              | .M="Missing"   1="Diagnostic CT"   2="Limited CT, but interpretable"   3="Non-diagnostic CT exam" |
| **ctdxqual_artifact**   | Reason for limited / non-diagnostic  CT: Severe beam hardening artifact |                                                              | .N="Not Applicable"   0="No"   1="Yes"                       |
| **ctdxqual_breath**     | Reason for limited / non-diagnostic  CT: Submaximal inspiratory breath-hold |                                                              | .N="Not Applicable"   0="No"   1="Yes"                       |
| **ctdxqual_graininess** | Reason for limited / non-diagnostic  CT: Excessive quantum mottle or graininess |                                                              | .N="Not Applicable"   0="No"   1="Yes"                       |
| **ctdxqual_inadeqimg**  | Reason for limited / non-diagnostic  CT: Lungs not completely imaged |                                                              | .N="Not Applicable"   0="No"   1="Yes"                       |
| **ctdxqual_motion**     | Reason for limited / non-diagnostic  CT: Motion artifact     |                                                              | .N="Not Applicable"   0="No"   1="Yes"                       |
| **ctdxqual_other**      | Reason for limited / non-diagnostic  CT: Other (specify)     |                                                              | .N="Not Applicable"   0="No"   1="Yes"                       |
| **ctdxqual_resp**       | Reason for limited / non-diagnostic  CT: Respiratory misregistration |                                                              | .N="Not Applicable"   0="No"   1="Yes"                       |
| **ctdxqual_techpara**   | Reason for limited / non-diagnostic  CT: Incorrect technical parameter(s) |                                                              | .N="Not Applicable"   0="No"   1="Yes"                       |
| **study_yr**            | Study Year of Screen                                         |                                                              | 0="T0"   1="T1"   2="T2"                                     |
| **techpara_effmas**     | Technical parameters: Effective mAs                          | Technical parameters: Effective mAs      This variable comes from the data collection forms. It may disagree with data extracted from  the CT images' DICOM headers. Header  data may be obtained from the NLST CT image collection at TCIA or from ACRIN. | Numeric   .M="Missing"                                       |
| **techpara_fov**        | Technical parameters: Display FOV in  cm                     | Technical parameters: Display FOV in  cm      This variable comes from the data collection forms.  It may disagree with data extracted from the  CT images' DICOM headers. Header data  may be obtained from the NLST CT image collection at TCIA or from ACRIN. | Numeric   .M="Missing"                                       |
| **techpara_kvp**        | Technical parameters: kVp                                    | Technical parameters: kVp      This variable comes from the data collection forms. It may disagree with data extracted from  the CT images' DICOM headers. Header  data may be obtained from the NLST CT image collection at TCIA or from ACRIN. | Numeric   .M="Missing"                                       |
| **techpara_ma**         | Technical parameters: mA                                     | Technical parameters: mA      This variable comes from the data collection forms. It may disagree with data extracted from  the CT images' DICOM headers. Header  data may be obtained from the NLST CT image collection at TCIA or from ACRIN. | Numeric   .M="Missing"                                       |

 

 

 

 

 