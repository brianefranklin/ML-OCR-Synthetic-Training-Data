#!/usr/bin/env python
import sys
import fontforge

def main(file):
    for font in fontforge.fontsInFile(file):
        f = fontforge.open(u'%s(%s)' % (file, font))
        f.generate('%s.ttf' % font)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('Usage: python ttc2ttf.py <input.ttc>')
    main(sys.argv[1])
