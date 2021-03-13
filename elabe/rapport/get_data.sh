

for y in {2015..2021};
do
for x in {1..10};
do
if ((($x) < 10));
then
m='0'$x
else
m=$x
fi
wget https://elabe.fr/wp-content/uploads/$y/$m/ -nd --recursive --accept '*baro*.pdf' --reject-regex '/\?C=[A-Z];O=[A-Z]$' ;
done
done