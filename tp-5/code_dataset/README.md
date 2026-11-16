To generate the code:
```
python3 ./generator.py
```

To generate the images
```
mkdir images
printf "%s\0" ../code/* | xargs -0 -n 1 -P 24 bash -c 'cat "$1" | ../Flex-Bison-Compiler "$(basename "$1").bmp"' _
```
