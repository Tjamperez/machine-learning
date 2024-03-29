import numpy as np
from collections import Counter

# Data
price = ['low', 'low', 'low', 'med', 'med', 'high', 'low', 'high', 'med', 'med', 'low', 'high', 'low', 'med', 'med', 'low', 'low', 'med', 'med', 'med', 'med', 'high', 'high', 'high', 'high']
lug_boot = ['med', 'med', 'small', 'small', 'big', 'small', 'big', 'med', 'med', 'big', 'big', 'big', 'small', 'med', 'big', 'med', 'small', 'med', 'med', 'small', 'big', 'big', 'big', 'small', 'small']
safety = ['med', 'high', 'med', 'med', 'high', 'high', 'high', 'med', 'med', 'med', 'med', 'high', 'med', 'high', 'low', 'low', 'low', 'med', 'low', 'low', 'med', 'med', 'low', 'low', 'med']
target = ['acc', 'acc', 'acc', 'acc', 'acc', 'acc', 'acc', 'acc', 'acc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc']

new_data = ['high','med','med']
pacct = 0 #total instances accepting car
punacct = 0#total instances of not accepting car
pacc = 0 #Probability of accepting car
punacc = 0#Probability of not accepting car
total = 0#Total instances of the data
pprc = 0#Chance of the target price being accepted
ppnrc = 0#Chance of the target price not being accepted
plbc = 0#Chance of the target lub_boot being accepted
pnlbc = 0#Chance of the target lug_boot not being accepted
psac = 0#Chance of the target safety being accepted
pnsac = 0#Chance of the target safety not being accepted
count = 0#Quantity of targets

total = len(target)

print(total)

for x in target:
    if x == 'acc':
        pacct += 1
    else:
        punacct += 1

print(pacct)
print(punacct)

pacc = pacct/total #

punacc = punacct/total

print(pacc)
print(punacc)

for x in price:
    if x == new_data[0] and count < pacct:
        pprc += 1
    elif x == new_data[0] and (count >= pacct and count < total):
        ppnrc += 1
    count += 1
pprc = pprc/pacct
ppnrc = ppnrc/punacct
count = 0

print(pprc)
print(ppnrc)

for x in lug_boot:
    if x == new_data[1] and count < pacct:
        plbc += 1
    elif x == new_data[1] and (count >= pacct and count < total):
        pnlbc += 1
    count += 1
plbc = plbc/pacct
pnlbc = pnlbc/punacct
count = 0

print(plbc)
print(pnlbc)

for x in safety:
    if x == new_data[2] and count < pacct:
        psac += 1
    elif x == new_data[2] and (count >= pacct and count < total):
        pnsac += 1
    count += 1
psac = psac/pacct
pnsac = pnsac/punacct
count = 0

print(psac)
print(pnsac)

#Probability of High Med Med Car being accepted
phmmacc = pacc*plbc*pprc*psac

#Probability of High Med Med Car being accepted
phmmunacc = punacc*pnlbc*ppnrc*pnsac

print(phmmacc)
print(phmmunacc)