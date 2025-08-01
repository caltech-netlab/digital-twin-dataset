# A Real-World Dataset
The dataset contains data collected from microgrids/distribution systems in California. The data has been anonymized to remove all geographical identifiers. Our [conference poster](assets/Poster.pdf) is available. A preprint is available on [arXiv](https://arxiv.org/abs/2504.06588).

## Sample dataset
Download sample dataset from [here](https://caltech.box.com/s/5baxy2ogbalqohpidh1lyxgnnxmv5tuc) and place into sample_dataset folder. 
Here, the data contains the same *types* of data in the complete dataset, but covering a very small time range of all availble data. This allows you to start developing your code for this dataset without dealing with the colossal full dataset (>10TB and growing). 

## Full dataset
The full dataset is hosted at [https://socal28bus.caltech.edu](https://socal28bus.caltech.edu) and can be accessed via a
REST API. Convenient access is provided by the Python class `DatasetApiClient` in
[dataset_api_client.py](dataset_api_client.py). See [Downloading data](#downloading-data)
for a code example.

Dataset users are authenticated using GitHub. Please [submit a ticket here] to have your
GitHub account added to the list of allowed users.

## Data types

In the following, we explain the 3 types of time-series data as well as circuit topology data in [`sample_dataset`](sample_dataset) folder:

### Magnitude
`magnitude` includes the root mean squared current and voltage magnitudes. These data do not contain phase angle information. The data is available at 1-second intervals.

### Synchro-phasor
`phasor` includes synchro-phasor measurements, which are presented as complex numbers. The data is available at 10-second intervals.

### Synchro-waveform
`waveform` includes the raw point-on-wave measurements sampled at 2.5kHz. Each waveform is roughly 1-second in length. One waveform is available every 10 seconds. The capture start times may differ by around 0.01 to 0.1 seconds. The sampling intervals are approximately 400 $\pm$ 4 microseconds.

Notice that the 3 types of data above are increasing in granularity. That is, given waveforms, one can compute phasors using Fast Fourier Transform (FFT). Given phasors, one can compute the magnitudes by taking the magnitude of the complex phasor values.

### Network and parameter
We also provide the time-varying circuit topolgy and parameters, which contain information such as line connectivity, transformer nameplate ratings, and circuit braker status. Importantly, the timeseries measurement metadata is also stored here.

We provide the most granular data which models the circuit down to individual components. Power transfer elements (e.g. lines, transformers, switches) are edges of the graph, whereas buses are nodes. This is called the **physical asset network**. The **electrical network** can be derived by zero- and infinite-impedance elements (e.g. short lines, closed/open breakers). Then nodes connected by zero-impedance elements are combined into a single node. 

The system model varies depending on your application. Examples include bus injection and branch flow models in the phasor domain, dynamic circuit model in the time domain, and transfer matrix models in the Laplace or z domains. See paper Section V for more detailed discussion.

<p align="center">
  <img src="assets/data_topology_w_meter.jpg" alt="Data Topology with Meter" width="600">
</p>

## Quickstart & code examples
### Setup
First, clone and `cd` into this repository.

Next, to run the following code examples, install the required Python packages in
[requirements.txt](requirements.txt) (e.g. `pip install -r requirements.txt`).

### Downloading data
The following Python code downloads the data, where:

- For `magnitudes_for`, `phasors_for`, and `waveforms_for`, `<element names>` should be
  replaced with network element names of interest.
- For `time_range`, `<start>` and `<end>` must be replaced by `datetime` objects, ISO 8601
  strings, or Unix timestamps.
- [Optional] For `resolution`, `<duration>` can be replaced with a `timedelta` object, a
  number of seconds, or an ISO 8601 duration string. If included, the data may be
  upsampled or downsampled.

```python
from dataset_api_client import DatasetApiClient

data_api_client = DatasetApiClient()
data_api_client.download_data(
    magnitudes_for=[<element names>],
    phasors_for=[<element names>],
    waveforms_for=[<element names>],
    time_range=(<start>, <end>),
    resolution=<duration>,
)
```

Here is an example query for magnitudes of egauge_1-CT1 for every minute throughout June
2024, using `datetime` and `timedelta` objects:

```python
from datetime import datetime, timedelta
from dataset_api_client import DatasetApiClient

data_api_client = DatasetApiClient()
data_api_client.download_data(
    magnitudes_for=["egauge_1-CT1"],
    time_range=(datetime(2024, 6, 1), datetime(2024, 7, 1)),
    resolution=timedelta(minutes=1),
)
```

> [!NOTE]  
> If there is no data for the selected element and time range, the API may return an
> empty file.

> [!WARNING]  
> The first time you run this code, it will prompt you to log in with GitHub and save the
> credentials in a file called `dataset_api_credentials.json`. Make sure to keep this file
> secret (e.g. do not share this file or commit it to a git repository).

Please [submit a ticket here] to have your GitHub account added to the list of allowed
users.

### Loading data
See example code in [`data_IO.ipynb`](code_examples/data_IO.ipynb).

### State estimation (synchro-phasor)
Given phasor measurements on a subset of nodes, we can recover the phasors for all network elements. This is described in paper Sections V (a) and VI (a). See example implementation in [`state_estimation_phasor.ipynb`](code_examples/state_estimation_phasor.ipynb).

### State estimation (synchro-waveform)
Given point on wave measurements and circuit parameters, we can simulate the time-domain circuit power flow. The formulation is described in paper Sections V (b) and VI (b). See example implementation in [`state_estimation_waveform.ipynb`](code_examples/state_estimation_waveform.ipynb).

### Voltage control
An example implementation of Linear DistFLow (LinDistFlow) model with measured slackbus voltage, real and reactive power injections along with a voltage control example is provided in [`voltage_control.ipynb`](code_examples/voltage_control.ipynb).

### MATPOWER Test Case
A test case based on the A side sub-circuit in [MATPOWER format](https://matpower.org/docs/ref/matpower5.0/caseformat.html) is provided in [`case12dt.m`](code_examples/case12dt.m)

### Support
If you think another example may be helpful here, consider contributing to this project via a pull request or [contact us](#contacts). In certain cases, we may be able to provide example code for your particular usage scenario.


## Data quality
As with any real-world dataset, inaccuracies, gaps, and unknown grid information are present in this dataset. We made a substatial effort to provide the highest quality data possible given practical constraints. Here, we make transparent where information may be inaccurate.

### Synchronization accuracy
From in-the-field testing, our meters show end-to-end synchronization error variance of 0.625 degrees. See paper section IV for a detailed discussion. The synchronization result is shown here. The phase angle is the difference between voltage phase angles of two meters measuring the same node.
<p align="center">
  <img src="assets/sync_test_result.png" width="600"/>
</p>

### Sensor error
Sensor error from meters as well as current and potential transformers are generally bounded by 0.5% (and typically even smaller in practice). One exception is where we have oversized current transformers in lightly loaded circuits. The large conductors require large current transformers (CTs) to fit around it. Large CTs generally have a higher current rating. In our deployment, channels `S1`, `S2`, `S3` in `egauge_18`, `egauge_20` are the only instances of this issue.

<p align="center">
  <img src="assets/oversized_CT.png" width="600"/>
</p>

### Gaps
Occasional gaps in data are present due to network and power outages, system maintenance. See section [Quickstart](#quickstart) for example code on handling data gaps.

### Circuit topology and parameter 
In practice, it is rare that distribution system operators maintain an error-free record of the system. 
- Lines: For distribution lines, the conductor thickness and material are obtained from engineering drawings and are generally correct, but the insulation material and thickness are estimated from popular cable types given the voltage level. Line lengths are estimated as the Manhattan (taxicab) distance between the two terminals. The lengths of short lines (within the same building structure) are assumed to be zero. Lines are generally underground, un-transposed with unknown cable arrangement (e.g. on a cable tray).
- Transformers: For transformers, the nameplate ratings are generally accurate. The true transformer tap positions are unknown, although off-nominal tap positions are rare. 
- Switches: The default status of `OpenClose` elements including switches, breakers, fuses, relays, etc. are documented in `.json` files. The switching activities during a power outage are obtained from system operators after the event and verified based on state estimation results to ensure accuracy.
- Grounding: Earth ground information are generally accurate, and grounding typically occurs on the secondary side of Delta-Wye transformers.
- Phase labels: In metered elements, `L1`, `L2`, `L3`, refer to phase voltages, `CT1`, `CT2`, `CT3` refer to current transformers, `S1`, `S2`, `S3` refer to meter current measurement channels. In general the numbers 1, 2, 3 correspond to phases A, B and C. Several exceptions are in `egauge_13`, `egauge_14`, `egauge_15`, `egauge_20`. When matching phases to measurements, use the information in `sample_dataset/topology/network_files/<path_to_file>.json` and `EgaugeMeter->registers->element`.

## Citation
BibTex:
```
@misc{xie2025dataset,
      title={A Digital Twin of an Electrical Distribution Grid: SoCal 28-Bus Dataset}, 
      author={Yiheng Xie and Lucien Werner and Kaibo Chen and Thuy-Linh Le and Christine Ortega and Steven Low},
      year={2025},
      eprint={2504.06588},
      archivePrefix={arXiv},
      primaryClass={eess.SY},
      url={https://arxiv.org/abs/2504.06588}, 
}
```
<!-- IEEE:
```
``` -->

## Contacts
We welcome your comments and suggestions at `digitaltwin@caltech.edu`. For discussions and improvements specific to code and data release, you may also use [GitHub Issues](https://github.com/caltech-netlab/digital-twin-dataset/issues) or [Pull requests](https://github.com/caltech-netlab/digital-twin-dataset/pulls).

## Disclaimer
The accuracy or reliability of the data is not guaranteed or warranted in any way and the providers disclaim liability of any kind whatsoever, including, without limitation, liability for quality, performance, merchantability and fitness for a particular purpose arising out of the use, or inability to use the data.

This software is provided by the copyright holders and contributors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the copyright owner or contributors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.

[submit a ticket here]: https://forms.office.com/r/Ds6rKEtyTV
