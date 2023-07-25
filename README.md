# Brushing-Teeth-Recognition

This code was used for my MS Thesis (<a href="https://oaktrust.library.tamu.edu/bitstream/handle/1969.1/161664/CHERIAN-THESIS-2017.pdf?sequence=1&isAllowed=y"><strong>Recognition of Everyday Activities through Wearable Sensors and Machine Learning</strong></a>) and a publication at Pervasive Health '17 (<a href="https://dl.acm.org/doi/10.1145/3397481.3450673"><strong>Did you remember to brush?: a noninvasive wearable approach to recognizing brushing teeth for elderly care</strong></a>).

For this work, we used accelerometer data collected from a Pebble smartwatch worn on the dominant hand. Data was collected from three different user studies ranging from a controlled lab study to a month-long in-the-wild study. Collected data were segmented into 4-second sliding windows (with 1-second overlap), and we extracted a combination of common and novel features from the windowed data. We used Weka for classification, the predicted classes were then filtered through a second Tier, which applied context-based post-processing to identify when individuals brushed their teeth with an <strong>accuracy of 94%</strong> and an <strong>F-measure of 0.82</strong>.

This code was written in 2015-2016, back when I was much more comfortable with Java than Python.

If you find this work useful, please cite our Pervasive Health paper.
## BibTeX

```bibtex
@inproceedings{cherian2017did,
    author = {Cherian, Josh and Rajanna, Vijay and Goldberg, Daniel and Hammond, Tracy},
    title = {Did You Remember to Brush? A Noninvasive Wearable Approach to Recognizing Brushing Teeth for Elderly Care},
    year = {2017},
    isbn = {9781450363631},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3154862.3154866},
    doi = {10.1145/3154862.3154866},
    abstract = {Failing to brush one's teeth regularly can have surprisingly serious health consequences, from periodontal disease to coronary heart disease to pancreatic cancer. This problem is especially worrying when caring for the elderly and/or individuals with dementia, as they often forget or are unable to perform standard health activities such as brushing their teeth, washing their hands, and taking medication. To ensure that such individuals are correctly looked after they are placed under the supervision of caretakers or family members, simultaneously limiting their independence and placing an immense burden on their family members and caretakers. To address this problem we developed a non-invasive wearable system based on a wrist-mounted accelerometer to accurately identify when a person brushed their teeth. We tested the efficacy of our system with a month-long in-the-wild study and achieved an accuracy of 94\% and an F-measure of 0.82.},
    booktitle = {Proceedings of the 11th EAI International Conference on Pervasive Computing Technologies for Healthcare},
    pages = {48–57},
    numpages = {10},
    keywords = {dementia, machine learning, elderly care, activity recognition, pervasive health, brushing teeth, wearable solution, intervention},
    location = {Barcelona, Spain},
    series = {PervasiveHealth '17}
}```
