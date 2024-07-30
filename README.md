# SU(2) gauge theory with one and two adjoint fermions towards the continuum limit

## Analysis workflow

This repository contains the analysis code used to prepare
the plots, tables, and other numbers used
in the publication
[SU(2) gauge theory with one and two adjoint fermions towards the continuum limit][arxiv].

It may be used to regenerate all of the above,
starting from raw data outputs from running on HPC,
plus data quoted from prior work.

## Setup

### Getting the code

If you are working from the Zenodo release of this workflow,
it is sufficient to download and extract the single archive.

If you are working from the GitHub repository,
then to download the workflow and its dependencies,
run

``` shellsession
$ git clone --recurse-submodules https://github.com/edbennett/su2_analysis
```

### Software environment

This workflow has been tested on macOS and Linux.
The authors are not aware of reasons it should not work on Windows,
but this is not tested.

Most dependencies for this workflow are managed via [Conda][conda].
If you do not already have Conda set up,
[Miniconda][miniconda] provides the quickest way to get set up.

Additionally,
LaTeX is required to be able to produce plots.
[TeX Live][texlive] is the best source for this.

With these prerequisites installed,
to create a Conda environment to run the workflow,
use

``` shellsession
$ conda env create -f environment.yml
```

(If the default name conflicts with one on your machine,
add `-n a_name_of_your_choice` to the command to override it.)

To activate this environment,
run

``` shellsession
$ conda activate su2
```

(where `su2` is replaced with any customised name you have set).

### Getting the data

The workflow depends on a number of sources of data:

1. Ensemble metadata.
   Download the file `ensembles.yaml` from [the data release][datarelease]
   and place it in the `metadata` directory.
2. Imported data.
   1. Download the file `Fig4.csv`
      from [the data release for the publication "Large mass hierarchies from strongly coupled dynamics"][largemass-dr],
      and place it in the `external_data` directory.
   2. Lattice data for $\mathrm{SU}(2)$,
      $N_{\mathrm{f}}=2$,
      $\beta=2.25$
      are included in this repository in the files
      `external_data/su2_nf2_b2.25.csv`
      and `external_data/su2_topology.csv`.
      No action is needed.
   3. Data output from the previous publication in this project.
      Download the file `su2.sqlite` from
      [the data release of the previous work][previous-dr]
      and place it in the `external_data` directory.
3. Raw data.
   1. Raw data from the previous publication in this project.
      Download the file `raw_data.zip` from
      [the data release of the previous work][previous-dr]
      and extract it into the root of the repository.
   2. New raw data prepared for this work.
      Download the file `raw_data.zip` from
      [the data release][datarelease]
      and extract it into the root of the repository.

## Running the workflow

With the data described above downloaded,
and the environment set up,
and the Conda environment active,
to run the full analysis end-to-end,
run

``` shellsession
# python -m autosu2 --ensembles metadata/ensembles.yaml --sideload_csv external_data/su2_nf2_b2.25.csv --sideload_sql external_data/su2.sqlite
```

This will output all plots, tables, and LaTeX definitions
included in [the paper][arxiv]
into the `assets` directory.
It will also output a CSV of final results
(also included in [the data release][datarelease])
into the `data` directory.

## Comments

This workflow is the product of seven years of gradual work,
and has been built along with the data that it analyses
and the research that it enables.
As such,
it is not necessarily designed to easily generalise to other data,
and its overall structure has many shortcomings
that would be designed differently were it written from scratch today.

However,
should any aspect of the workflow prove useful in your work,
please do borrow it,
subject to the terms in `LICENSE`,
and making sure to cite this work as specified in `CITATION.cff`.

[arxiv]: TODO
[conda]: https://anaconda.org/anaconda/conda
[datarelease]: https://doi.org/10.5281/zenodo.13128505
[largemass-dr]: https://doi.org/10.5281/zenodo.13128485
[miniconda]: https://docs.anaconda.com/miniconda/
[previous-dr]: https://doi.org/10.5281/zenodo.5139618
[texlive]: https://www.tug.org/texlive/
