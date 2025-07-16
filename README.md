# Covid_severity_prediction

Model to predict if a patient will develop severe COVID-19, defined as death or the need for intubation within one month from the acquisition of the baseline CT scan.
The prediction is made based on features derived from the CT scan and the patient's age and sex.
## Usage
All inputs should be provided in the _input_ folder according to: <br>
- /input/patient_id/CT.nii.gz <br>
- /input/demographic_data.csv
Two example patients are given in demographic_data.csv to demonstrate the format.
## Reference
For more details see our [publication](https://doi.org/10.1186/s12911-025-02983-z). If you use this tool please cite it as follows:
<br>
Dirks, I., Bossa, M., Berenguer, A. et al. Development and multicentric external validation of a prognostic COVID-19 severity model based on thoracic CT. BMC Med Inform Decis Mak 25, 156 (2025). https://doi.org/10.1186/s12911-025-02983-z
