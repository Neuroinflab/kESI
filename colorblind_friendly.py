# Based on
# Color Universal Design (CUD)
# - How to make figures and presentations that are friendly to Colorblind people
#
#
# Masataka Okabe
# Jikei Medial School (Japan)
#
# Kei Ito
# University of Tokyo, Institute for Molecular and Cellular Biosciences (Japan)
# (both are strong protanopes)
# 11.20.2002 (modified on 2.15.2008, 9.24.2008)
# http://jfly.iam.u-tokyo.ac.jp/color/#pallet

def _html(r, g, b):
    return "#{:02X}{:02X}{:02X}".format(r, g, b)

BLACK     = _html(  0,   0,   0)
ORANGE    = _html(230, 159,   0)
SKY_BLUE  = _html( 86, 180, 233)
GREEN     = _html(  0, 158, 115)
YELLOW    = _html(240, 228,  66)
BLUE      = _html(  0, 114, 178)
VERMILION = _html(213,  94,   0)
PURPLE    = _html(204, 121, 167)