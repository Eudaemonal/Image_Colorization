
# Image Colorization


## Getting Started


### Prerequisites


#### Building training environment

Putting the following in one folder

```
* train.py
* resize.py
* images/...
* resized/...
* model/vgg16-20160129.tfmodel
```

The images/... folder will contain .jpg images for training, it can be any color image that you downloaded.

The resized/... folder will be an empty folder to store resized image for training

The model/... folder will contain the file vgg16-20160129.tfmodel, which can be found on google

### Training

#### Resizing

```bash
python3 -m dataset.resize <args>
```

Use `-h` to see the available options

In our example

```bash
python3 -m dataset.resize -s images -o resized
```

#### Training

Train the neural network

```bash
python3 train.py
```


## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc