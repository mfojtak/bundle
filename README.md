# bundle
Bundles python project into dockerized kubernetes app

## Install
```bash
pip install git+https://github.com/mfojtak/bundle.git
```

## Usage
Build and apply bundle
```bash
bundlectl apply <project folder> <config file>.yaml
```

Build only
```bash
bundlectl build <project folder> <config file>.yaml
```
