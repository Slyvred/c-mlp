#!/bin/bash

gcc -O3 -Wall -o main main.c helpers.c


--> déploiement
- vendredi soir
- instable en prod
- plaintes des clients
- 'marche chez eux'

1. qui intervient
- l'équipe dev / potentiellement les devops qui ont foirés
- ordre: dev / devops

2.
- déploiement un vendredi soir --> mauvaise planification
- manque de tests
- partie CI trop superficielle/baclée --> aurait du détecter les problèmes
- frictions:
  - manque de supervision et de planification de la part de la direction
- flous:
  - équipe dev mais qui s'occupe du monitoring ? du déploiement ?
- risques:
  - régression
  - pertes économiques
  -
