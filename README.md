```
saumikdana@Saumiks-Laptop Bayesian-Markov-chain-Monte-Carlo % mkdir docs

saumikdana@Saumiks-Laptop Bayesian-Markov-chain-Monte-Carlo % cd docs

saumikdana@Saumiks-Laptop docs % /usr/bin/python3 -m sphinx.cmd.quickstart

saumikdana@Saumiks-Laptop docs % /usr/bin/python3 -m sphinx.ext.apidoc -o source .. --force

saumikdana@Saumiks-Laptop docs % /usr/bin/python3 -m sphinx -b html source build
```