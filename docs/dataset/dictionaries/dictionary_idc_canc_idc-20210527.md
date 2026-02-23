# IDC: Lung Cancer: Data Dictionary

 

## Table of Contents

 

| Section Number | Section Title |
| -------------- | ------------- |
| 1              | Study         |
| 2              | Lung Cancer   |

 

 

 

 

 

 



 

 



# Document Summary

 

| Property          | Value                                |
| ----------------- | ------------------------------------ |
| Document Title    | IDC: Lung Cancer: Data Dictionary    |
| Date Created      | 05/28/2021                           |
| Sections          | 2                                    |
| Entries           | 34                                   |
| Document Filename | dictionary_idc_canc_idc-20210527.rtf |

 

 

 

 

 

 



 

 



# IDC: Lung Cancer: Data Dictionary

 

## Section 1: Study

 

| Variable            | Label                   | Description                                                  | Format Text |
| ------------------- | ----------------------- | ------------------------------------------------------------ | ----------- |
| **dataset_version** | Date Stamp for Datasets |                                                              | Char, 23    |
| **pid**             | Participant Identifier  | A unique identifier given to each  participant. For LSS participants, pid  has a format of 1xx,xxx, while for ACRIN participants, pid has a format of  2xx,xxx. | Numeric     |

 

 

 

 

 

 



 

 



## Section 2: Lung Cancer

 

| Variable              | Label                                                        | Description                                                  | Format Text                                                  |
| --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **candx_days**        | Days from randomization to diagnosis  of lung cancer         |                                                              | Numeric                                                      |
| **clinical_m**        | Clinical M for staging                                       | Clinical M code for staging (AJCC 6).                        | .M="Missing"   0="M0"   100="M1"   999="MX"                  |
| **clinical_m_7thed**  | AJCC 7th edition staging clinical M  component               |                                                              | .M="Missing"   0="M0"   100="M1"   110="M1a"   120="M1b"   999="MX" |
| **clinical_n**        | Clinical N for staging                                       | Clinical N code for staging (AJCC 6).                        | .M="Missing"   0="N0"   100="N1"   200="N2"   300="N3"   999="NX" |
| **clinical_n_7thed**  | AJCC 7th edition staging clinical N  component               |                                                              | .M="Missing"   0="N0"   100="N1"   200="N2"   300="N3"   999="NX" |
| **clinical_stag**     | Clinical Stage                                               | Clinical stage of lung cancer (AJCC  6).                     | .M="Missing"   110="Stage IA"   120="Stage IB"   210="Stage IIA"   220="Stage IIB"   310="Stage IIIA"   320="Stage IIIB"   400="Stage IV"   888="TNM not available"   900="Occult Carcinoma"   994="Carcinoid, cannot be assessed"   999="Unknown, cannot be assessed" |
| **clinical_t**        | Clinical T for staging                                       | Clinical T code for staging (AJCC 6).                        | .M="Missing"   100="T1"   200="T2"   300="T3"   400="T4"   999="TX" |
| **clinical_t_7thed**  | AJCC 7th edition staging clinical T  component               |                                                              | .M="Missing"   110="T1a"   120="T1b"   200="T2"   210="T2a"   220="T2b"   300="T3"   400="T4"   999="TX" |
| **de_grade**          | Lung Cancer Grade                                            | Lung cancer grade. For ACRIN, this is the ICD-O-3 grade. For LSS, this comes from a separate  question on the DE form. | 1="Grade Cannot be  Assessed"   2="Well Differentiated (G1)"   3="Moderately Differentiated (G2)"   4="Poorly Differentiated (G3)"   5="Undifferentiated (G4)"   6="Unspecified in Pathology Report"   8="Unknown"   9="Missing" |
| **de_stag**           | Stage ("Best": Path if  avail., else Clin)                   | Lung cancer stage (AJCC 6), combining  clinical and pathologic staging information. | .M="Missing"   110="Stage IA"   120="Stage IB"   210="Stage IIA"   220="Stage IIB"   310="Stage IIIA"   320="Stage IIIB"   400="Stage IV"   888="TNM not available"   900="Occult Carcinoma"   994="Carcinoid, cannot be assessed"   999="Unknown, cannot be assessed" |
| **de_stag_7thed**     | AJCC 7th edition stage                                       | Stage of first primary lung cancer,  based on AJCC 7th edition stage | .M="Missing"   110="Stage IA"   120="Stage IB"   210="Stage IIA"   220="Stage IIB"   310="Stage IIIA"   320="Stage IIIB"   400="Stage IV"   888="TNM not available"   900="Occult Carcinoma"   999="Unknown, cannot be assessed" |
| **de_type**           | ICD-O-3 Morphology (from histology)                          | Lung cancer type from ICD-O-3  morphology. For LSS participants, this  is recorded separately from the complete ICD code, and represents the best  information available on the type of the cancer. | Numeric   .M="Missing"                                       |
| **first_lc**          | Is this cancer the first lung cancer  diagnosed?             | Is this cancer the first lung cancer  diagnosed? Participants with multiple  primary lung cancers will have a separate record included in this dataset for  each cancer. | 0="No"   1="Yes"                                             |
| **lc_behav**          | ICD-O-3 Behavior                                             | ICD-O-3 behavior of lung cancer.                             | 1="Borderline Malignancy"   3="Invasive"   6="Metastatic"    |
| **lc_grade**          | ICD-O-3 Grade                                                | ICD-O-3 grade of lung cancer.                                | 1="Well Differentiated: Grade  I"   2="Moderately Differentiated: Grade II"   3="Poorly Differentiated; Grade III"   4="Undifferentiated; Grade IV"   9="Unknown" |
| **lc_morph**          | ICD-O-3 Morphology                                           | ICD-O-3 morphology of lung cancer.                           | Numeric                                                      |
| **lc_order**          | Order of this lung cancer among all  lung cancers for this participant | The order of this lung cancer among  all lung cancers for this participant.   Order is from earliest to latest. | Numeric                                                      |
| **lc_topog**          | ICD-O-3 Topography                                           | ICD-O-3 topography of lung cancer.                           | Char, 5                                                      |
| **lesionsize**        | Tumor size (mm) - Pathology                                  |                                                              | Numeric   .M="Missing"                                       |
| **path_m**            | Pathologic M for staging                                     | Pathologic M code for staging (AJCC  6).                     | .M="Missing"   0="M0"   100="M1"   999="MX"                  |
| **path_m_7thed**      | AJCC 7th edition staging path M  component                   |                                                              | .M="Missing"   0="M0"   100="M1"   110="M1a"   120="M1b"   999="MX" |
| **path_n**            | Pathologic N for staging                                     | Pathologic N code for staging (AJCC  6).                     | .M="Missing"   0="N0"   100="N1"   200="N2"   300="N3"   999="NX" |
| **path_n_7thed**      | AJCC 7th edition staging path N  component                   |                                                              | .M="Missing"   0="N0"   100="N1"   200="N2"   300="N3"   999="NX" |
| **path_stag**         | Pathologic Stage                                             | Pathologic stage of lung cancer (AJCC  6).                   | .M="Missing"   110="Stage IA"   120="Stage IB"   210="Stage IIA"   220="Stage IIB"   310="Stage IIIA"   320="Stage IIIB"   400="Stage IV"   888="TNM not available"   900="Occult Carcinoma"   994="Carcinoid, cannot be assessed"   999="Unknown, cannot be assessed" |
| **path_t**            | Pathologic T for staging                                     | Pathologic T code for staging (AJCC  6).                     | .M="Missing"   100="T1"   200="T2"   300="T3"   400="T4"   999="TX" |
| **path_t_7thed**      | AJCC 7th edition staging path T  component                   |                                                              | .M="Missing"   110="T1a"   120="T1b"   200="T2"   210="T2a"   220="T2b"   300="T3"   400="T4"   999="TX" |
| **source_best_stage** | Source of "best" stage  (de_stag)                            | Describes whether the TNM components  used to make the "best" stage variable (de_stag) were pathologic,  clinical, a combination, or unavailable for some reason. | 1="Pathological"   2="Clinical"   3="Mixture"   5="Reporting stage only"   6="Stage cannot be assessed"   94="Carcinoid, stage cannot be assessed"   98="TNM not available"   99="Missing TNM" |
| **stage_only**        | Stage Only (if separate T, N, & M  not available)            | Provides stage when separate TNM  information is not available.    For all ACRIN lung cancers, this question was completed, i.e. not  missing.    For LSS lung cancers, this question was not expected to be completed unless  T/N/M components of pathologic stage were unavailable. | .M="Missing"   110="Stage IA"   120="Stage IB"   210="Stage IIA"   220="Stage IIB"   310="Stage IIIA"   320="Stage IIIB"   400="Stage IV"   888="TNM not available"   910="No evidence of tumor" |
| **stage_sum**         | Summary staging                                              | Summary staging.   For all ACRIN lung cancers, this question was completed, i.e. not missing.   For LSS lung cancers, this question was not expected to be completed unless  T/N/M components of pathologic stage were unavailable. | .M="Missing"   1="Localized"   2="Regional"   3="Distant"   4="Not available" |
| **study_yr**          | Study Year of Diagnosis                                      |                                                              | 0="T0"   1="T1"   2="T2"   3="T3"   4="T4"   5="T5"   6="T6"   7="T7" |
| **topog_source**      | Source of samples for ICD-O-3 code                           | Indicates the source of information  used to determine the ICD-O-3 code. | .M="Missing"   1="Cytology"   2="Histology"   3="Combined"   4="Clinical (LSS only)" |
| **valcsg**            | VALCSG Stage (Small cell only)                               | VALCSG staging for small cell lung  cancers.   For all ACRIN lung cancers, this question was completed, i.e. not missing,  (even if they were not small cell cancers). | .M="Missing"   1="Limited"   2="Extensive"   3="Not available" |

 

 

 

 

 