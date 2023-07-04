# graphtempo-triangles
Extended version of GraphTempo paper implementing structure aggregation

# graphtempo-triangles

The code repository for the following paper:

## Exploring....

Evangelia Tsoukanara, Georgia Koloniari, and Evaggelia Pitoura. Skyline-based Termporal Graph Exploration.

## Abstract
> 

## General Information
This repository facilitates pattern exploration on temporal attributed graphs, by detecting significant events, such as _stability_, _shrinkage_, and _growth_, w.r.t. specific structures / patterns in the graph. The datasets used in this paper is provided in `datasets`.

## Datasets
_DBLP_: directed collaboration dataset that spans over a period of 21 years (2000 to 2020) and includes publicatoins at 21 conferences related to data management research areas. Each node corresponds to an author and is associated with one static (gender), and one time-varying (#publications) attribute.

_MovieLens_: directed mutual rating dataset (built on the benchmark movie ratings dataset) covering a period of six months (May 1st, 2000 to October 31st, 2000) where each node represents a user and an edge denotes that two users have rated the same movie, and is attributed with three static (gender, age, occupation) and one time-varying attribute (average rating per month).

_Primary School_: undirected face-to-face proximity network describing the interactions between students and teachers at a primary school of Lyon, France. The dataset covers a period of 17 hours, each node is associated with two static attributes, gender, and class.

## Dependencies
Python 3.7
