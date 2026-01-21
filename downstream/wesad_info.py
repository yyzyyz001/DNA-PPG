SAMPLING_RATE = 64  # Hz

wesad_s2_info = {
    'base': {
        'valence': 7,
        'arousal': 2,
        'start_sec': 7*60 + 8,      
        'end_sec': 26*60 + 32,      
        'start_idx': (7*60 + 8) * SAMPLING_RATE,
        'end_idx': (26*60 + 32) * SAMPLING_RATE
    },
    'tsst': {
        'valence': 5,
        'arousal': 4,
        'start_sec': 39*60 + 55,    
        'end_sec': 50*60 + 30,      
        'start_idx': (39*60 + 55) * SAMPLING_RATE,
        'end_idx': (50*60 + 30) * SAMPLING_RATE
    },
    'medi1': {
        'valence': 7,
        'arousal': 2,
        'start_sec': 70*60 + 19,    
        'end_sec': 77*60 + 10,      
        'start_idx': (70*60 + 19) * SAMPLING_RATE,
        'end_idx': (77*60 + 10) * SAMPLING_RATE
    },
    'fun': {
        'valence': 8,
        'arousal': 1,
        'start_sec': 81*60 + 25,    
        'end_sec': 87*60 + 47,      
        'start_idx': (81*60 + 25) * SAMPLING_RATE,
        'end_idx': (87*60 + 47) * SAMPLING_RATE
    },
    'medi2': {
        'valence': 7,
        'arousal': 2,
        'start_sec': 93*60 + 38,    
        'end_sec': 100*60 + 15,     
        'start_idx': (93*60 + 38) * SAMPLING_RATE,
        'end_idx': (100*60 + 15) * SAMPLING_RATE
    }
}


wesad_s3_info = {
    'base': {
        'valence': 7,
        'arousal': 4,
        'start_sec': 6*60 + 44,    # 6.44 → 404 s
        'end_sec': 26*60 + 4,      # 26.04 → 1564 s
        'start_idx': (6*60 + 44) * SAMPLING_RATE,   # 404*64 = 25856
        'end_idx': (26*60 + 4) * SAMPLING_RATE      # 1564*64 = 100096
    },
    'tsst': {
        'valence': 7,
        'arousal': 7,
        'start_sec': 38*60 + 15,   # 2295 s
        'end_sec': 49*60 + 15,     # 2955 s
        'start_idx': (38*60 + 15) * SAMPLING_RATE,  # 146880
        'end_idx': (49*60 + 15) * SAMPLING_RATE     # 189120
    },
    'medi1': {
        'valence': 6,
        'arousal': 3,
        'start_sec': 72*60 + 28,   # 4348 s
        'end_sec': 79*60 + 30,     # 4770 s
        'start_idx': (72*60 + 28) * SAMPLING_RATE,  # 278272
        'end_idx': (79*60 + 30) * SAMPLING_RATE     # 305280
    },
    'fun': {
        'valence': 8,
        'arousal': 4,
        'start_sec': 84*60 + 15,   # 5055 s
        'end_sec': 90*60 + 50,     # 5450 s
        'start_idx': (84*60 + 15) * SAMPLING_RATE,  # 323520
        'end_idx': (90*60 + 50) * SAMPLING_RATE     # 348800
    },
    'medi2': {
        'valence': 6,
        'arousal': 3,
        'start_sec': 97*60 + 49,   # 5869 s
        'end_sec': 104*60 + 27,    # 6267 s
        'start_idx': (97*60 + 49) * SAMPLING_RATE,  # 375616
        'end_idx': (104*60 + 27) * SAMPLING_RATE    # 401088
    }
}


wesad_s4_info = {
    'base': {
        'valence': 7,
        'arousal': 2,
        'start_sec': 5*60 + 52,     # 5.52 -> 352s
        'end_sec': 25*60 + 30,      # 25.30 -> 1530s
        'start_idx': (5*60 + 52) * SAMPLING_RATE,
        'end_idx': (25*60 + 30) * SAMPLING_RATE
    },
    'fun': {
        'valence': 8,
        'arousal': 1,
        'start_sec': 31*60 + 39,    # 31.39 -> 1899s
        'end_sec': 38*60 + 11,      # 38.11 -> 2291s
        'start_idx': (31*60 + 39) * SAMPLING_RATE,
        'end_idx': (38*60 + 11) * SAMPLING_RATE
    },
    'medi1': {
        'valence': 5,
        'arousal': 7,
        'start_sec': 45*60 + 52,    # 45.52 -> 2752s
        'end_sec': 53*60 + 0,       # 53.00 -> 3180s
        'start_idx': (45*60 + 52) * SAMPLING_RATE,
        'end_idx': (53*60 + 0) * SAMPLING_RATE
    },
    'tsst': {
        'valence': 5,
        'arousal': 7,
        'start_sec': 61*60 + 20,    # 61.20 -> 3680s
        'end_sec': 72*60 + 15,      # 72.15 -> 4335s
        'start_idx': (61*60 + 20) * SAMPLING_RATE,
        'end_idx': (72*60 + 15) * SAMPLING_RATE
    },
    'medi2': {
        'valence': 5,
        'arousal': 2,
        'start_sec': 95*60 + 43,    # 95.43 -> 5743s
        'end_sec': 102*60 + 40,     # 102.40 -> 6160s
        'start_idx': (95*60 + 43) * SAMPLING_RATE,
        'end_idx': (102*60 + 40) * SAMPLING_RATE
    }
}


wesad_s5_info = {
    'base': {
        'valence': 7,
        'arousal': 2,
        'start_sec': 5*60 + 37,     # 5.37 -> 337 s
        'end_sec': 25*60 + 55,      # 25.55 -> 1555 s
        'start_idx': (5*60 + 37) * SAMPLING_RATE,
        'end_idx': (25*60 + 55) * SAMPLING_RATE
    },
    'fun': {
        'valence': 6,
        'arousal': 2,
        'start_sec': 32*60 + 0,     # 32.00 -> 1920 s
        'end_sec': 38*60 + 34,      # 38.34 -> 2314 s
        'start_idx': (32*60 + 0) * SAMPLING_RATE,
        'end_idx': (38*60 + 34) * SAMPLING_RATE
    },
    'medi1': {
        'valence': 6,
        'arousal': 1,
        'start_sec': 45*60 + 43,    # 45.43 -> 2743 s
        'end_sec': 52*60 + 40,      # 52.40 -> 3160 s
        'start_idx': (45*60 + 43) * SAMPLING_RATE,
        'end_idx': (52*60 + 40) * SAMPLING_RATE
    },
    'tsst': {
        'valence': 5,
        'arousal': 8,
        'start_sec': 61*60 + 0,     # 61.00 -> 3660 s
        'end_sec': 72*60 + 5,       # 72.05 -> 4325 s
        'start_idx': (61*60 + 0) * SAMPLING_RATE,
        'end_idx': (72*60 + 5) * SAMPLING_RATE
    },
    'medi2': {
        'valence': 7,
        'arousal': 1,
        'start_sec': 92*60 + 15,    # 92.15 -> 5535 s
        'end_sec': 99*60 + 12,      # 99.12 -> 5952 s
        'start_idx': (92*60 + 15) * SAMPLING_RATE,
        'end_idx': (99*60 + 12) * SAMPLING_RATE
    }
}


wesad_s6_info = {
    'base': {
        'valence': 8,
        'arousal': 2,
        'start_sec': 11*60 + 17,   # 11.17 -> 677s
        'end_sec': 31*60 + 17,     # 31.17 -> 1877s
        'start_idx': (11*60 + 17) * SAMPLING_RATE,
        'end_idx': (31*60 + 17) * SAMPLING_RATE
    },
    'tsst': {
        'valence': 6,
        'arousal': 3,
        'start_sec': 41*60 + 15,   # 41.15 -> 2475s
        'end_sec': 52*60 + 25,     # 52.25 -> 3145s
        'start_idx': (41*60 + 15) * SAMPLING_RATE,
        'end_idx': (52*60 + 25) * SAMPLING_RATE
    },
    'medi1': {
        'valence': 6,
        'arousal': 2,
        'start_sec': 75*60 + 54,   # 75.54 -> 4554s
        'end_sec': 82*60 + 51,     # 82.51 -> 4971s
        'start_idx': (75*60 + 54) * SAMPLING_RATE,
        'end_idx': (82*60 + 51) * SAMPLING_RATE
    },
    'fun': {
        'valence': 8,
        'arousal': 2,
        'start_sec': 93*60 + 5,    # 93.05 -> 5585s
        'end_sec': 99*60 + 37,     # 99.37 -> 5977s
        'start_idx': (93*60 + 5) * SAMPLING_RATE,
        'end_idx': (99*60 + 37) * SAMPLING_RATE
    },
    'medi2': {
        'valence': 8,
        'arousal': 1,
        'start_sec': 109*60 + 0,   # 109 -> 6540s
        'end_sec': 115*60 + 50,    # 115.50 -> 6950s
        'start_idx': (109*60 + 0) * SAMPLING_RATE,
        'end_idx': (115*60 + 50) * SAMPLING_RATE
    }
}


wesad_s7_info = {
    'base': {
        'valence': 7,
        'arousal': 2,
        'start_sec': 2*60 + 14,     # 2.14 → 134 s
        'end_sec': 22*60 + 20,      # 22.20 → 1340 s
        'start_idx': (2*60 + 14) * SAMPLING_RATE,
        'end_idx': (22*60 + 20) * SAMPLING_RATE
    },
    'fun': {
        'valence': 7,
        'arousal': 7,
        'start_sec': 26*60 + 24,    # 26.24 → 1584 s
        'end_sec': 32*60 + 56,      # 32.56 → 1976 s
        'start_idx': (26*60 + 24) * SAMPLING_RATE,
        'end_idx': (32*60 + 56) * SAMPLING_RATE
    },
    'medi1': {
        'valence': 7,
        'arousal': 2,
        'start_sec': 37*60 + 21,    # 37.21 → 2241 s
        'end_sec': 44*60 + 18,      # 44.18 → 2658 s
        'start_idx': (37*60 + 21) * SAMPLING_RATE,
        'end_idx': (44*60 + 18) * SAMPLING_RATE
    },
    'tsst': {
        'valence': 6,
        'arousal': 7,
        'start_sec': 52*60 + 10,    # 52.10 → 3130 s
        'end_sec': 63*60 + 10,      # 63.10 → 3790 s
        'start_idx': (52*60 + 10) * SAMPLING_RATE,
        'end_idx': (63*60 + 10) * SAMPLING_RATE
    },
    'medi2': {
        'valence': 7,
        'arousal': 3,
        'start_sec': 78*60 + 7,     # 78.07 → 4687 s
        'end_sec': 85*60 + 0,       # 85.00 → 5100 s
        'start_idx': (78*60 + 7) * SAMPLING_RATE,
        'end_idx': (85*60 + 0) * SAMPLING_RATE
    }
}


wesad_s8_info = {
    'base': {
        'valence': 7,
        'arousal': 3,
        'start_sec': 3*60 + 56,   # 3.56 → 236 s
        'end_sec': 23*60 + 45,    # 23.45 → 1425 s
        'start_idx': (3*60 + 56) * SAMPLING_RATE,   # 15104
        'end_idx': (23*60 + 45) * SAMPLING_RATE     # 91200
    },
    'fun': {
        'valence': 7,
        'arousal': 3,
        'start_sec': 29*60 + 0,   # 29.00 → 1740 s
        'end_sec': 35*60 + 30,    # 35.30 → 2130 s
        'start_idx': (29*60 + 0) * SAMPLING_RATE,   # 111360
        'end_idx': (35*60 + 30) * SAMPLING_RATE     # 136320
    },
    'medi1': {
        'valence': 7,
        'arousal': 2,
        'start_sec': 39*60 + 52,  # 39.52 → 2392 s
        'end_sec': 46*60 + 52,    # 46.52 → 2812 s
        'start_idx': (39*60 + 52) * SAMPLING_RATE,  # 153088
        'end_idx': (46*60 + 52) * SAMPLING_RATE     # 179968
    },
    'tsst': {
        'valence': 5,
        'arousal': 7,
        'start_sec': 56*60 + 10,  # 56.10 → 3370 s
        'end_sec': 67*60 + 40,    # 67.40 → 4060 s
        'start_idx': (56*60 + 10) * SAMPLING_RATE,  # 215680
        'end_idx': (67*60 + 40) * SAMPLING_RATE     # 259840
    },
    'medi2': {
        'valence': 6,
        'arousal': 4,
        'start_sec': 82*60 + 54,  # 82.54 → 4974 s
        'end_sec': 89*60 + 50,    # 89.50 → 5390 s
        'start_idx': (82*60 + 54) * SAMPLING_RATE,  # 318336
        'end_idx': (89*60 + 50) * SAMPLING_RATE     # 344960
    }
}


wesad_s9_info = {
    'base': {
        'valence': 8,
        'arousal': 3,
        'start_sec': 1*60 + 48,     # 1.48 -> 108 s
        'end_sec': 21*60 + 48,      # 21.48 -> 1308 s
        'start_idx': (1*60 + 48) * SAMPLING_RATE,
        'end_idx': (21*60 + 48) * SAMPLING_RATE
    },
    'tsst': {
        'valence': 7,
        'arousal': 4,
        'start_sec': 31*60 + 45,    # 31.45 -> 1905 s
        'end_sec': 42*60 + 50,      # 42.50 -> 2570 s
        'start_idx': (31*60 + 45) * SAMPLING_RATE,
        'end_idx': (42*60 + 50) * SAMPLING_RATE
    },
    'medi1': {
        'valence': 9,
        'arousal': 1,
        'start_sec': 59*60 + 2,     # 59.02 -> 3542 s
        'end_sec': 66*60 + 0,       # 66.00 -> 3960 s
        'start_idx': (59*60 + 2) * SAMPLING_RATE,
        'end_idx': (66*60 + 0) * SAMPLING_RATE
    },
    'fun': {
        'valence': 7,
        'arousal': 3,
        'start_sec': 68*60 + 11,    # 68.11 -> 4091 s
        'end_sec': 74*60 + 43,      # 74.43 -> 4483 s
        'start_idx': (68*60 + 11) * SAMPLING_RATE,
        'end_idx': (74*60 + 43) * SAMPLING_RATE
    },
    'medi2': {
        'valence': 9,
        'arousal': 1,
        'start_sec': 78*60 + 45,    # 78.45 -> 4725 s
        'end_sec': 85*60 + 40,      # 85.40 -> 5140 s
        'start_idx': (78*60 + 45) * SAMPLING_RATE,
        'end_idx': (85*60 + 40) * SAMPLING_RATE
    }
}


wesad_s10_info = {
    'base': {
        'valence': 6,
        'arousal': 2,
        'start_sec': 2*60 + 5,      # 2.5 -> 125 s
        'end_sec': 22*60 + 5,       # 22.5 -> 1325 s
        'start_idx': (2*60 + 5) * SAMPLING_RATE,
        'end_idx': (22*60 + 5) * SAMPLING_RATE
    },
    'fun': {
        'valence': 8,
        'arousal': 2,
        'start_sec': 27*60 + 53,    # 27.53 -> 1673 s
        'end_sec': 34*60 + 25,      # 34.25 -> 2065 s
        'start_idx': (27*60 + 53) * SAMPLING_RATE,
        'end_idx': (34*60 + 25) * SAMPLING_RATE
    },
    'medi1': {
        'valence': 6,
        'arousal': 1,
        'start_sec': 38*60 + 42,    # 38.42 -> 2322 s
        'end_sec': 45*60 + 40,      # 45.4 -> 2740 s
        'start_idx': (38*60 + 42) * SAMPLING_RATE,
        'end_idx': (45*60 + 40) * SAMPLING_RATE
    },
    'tsst': {
        'valence': 3,
        'arousal': 8,
        'start_sec': 54*60 + 30,    # 54.3 -> 3270 s
        'end_sec': 66*60 + 55,      # 66.55 -> 4015 s
        'start_idx': (54*60 + 30) * SAMPLING_RATE,
        'end_idx': (66*60 + 55) * SAMPLING_RATE
    },
    'medi2': {
        'valence': 6,
        'arousal': 2,
        'start_sec': 83*60 + 17,    # 83.17 -> 4997 s
        'end_sec': 90*60 + 15,      # 90.15 -> 5415 s
        'start_idx': (83*60 + 17) * SAMPLING_RATE,
        'end_idx': (90*60 + 15) * SAMPLING_RATE
    }
}


wesad_s11_info = {
    'base': {
        'valence': 6,
        'arousal': 2,
        'start_sec': 2*60 + 23,   # 143 s
        'end_sec': 22*60 + 23,    # 1343 s
        'start_idx': (2*60 + 23) * SAMPLING_RATE,  # 9152
        'end_idx': (22*60 + 23) * SAMPLING_RATE    # 85952
    },
    'tsst': {
        'valence': 4,
        'arousal': 6,
        'start_sec': 32*60 + 45,  # 1965 s
        'end_sec': 44*60 + 25,    # 2665 s
        'start_idx': (32*60 + 45) * SAMPLING_RATE, # 125760
        'end_idx': (44*60 + 25) * SAMPLING_RATE    # 170560
    },
    'medi1': {
        'valence': 7,
        'arousal': 2,
        'start_sec': 60*60 + 30,  # 3630 s
        'end_sec': 67*60 + 25,    # 4045 s
        'start_idx': (60*60 + 30) * SAMPLING_RATE, # 232320
        'end_idx': (67*60 + 25) * SAMPLING_RATE    # 258880
    },
    'fun': {
        'valence': 8,
        'arousal': 3,
        'start_sec': 69*60 + 27,  # 4167 s
        'end_sec': 75*60 + 55,    # 4555 s
        'start_idx': (69*60 + 27) * SAMPLING_RATE, # 266688
        'end_idx': (75*60 + 55) * SAMPLING_RATE    # 291520
    },
    'medi2': {
        'valence': 6,
        'arousal': 2,
        'start_sec': 79*60 + 9,   # 4749 s
        'end_sec': 86*60 + 5,     # 5165 s
        'start_idx': (79*60 + 9) * SAMPLING_RATE,  # 303936
        'end_idx': (86*60 + 5) * SAMPLING_RATE     # 330560
    }
}


wesad_s13_info = {
    'base': {
        'valence': 5,
        'arousal': 3,
        'start_sec': 2*60 + 28,       # 2.28 -> 148 s
        'end_sec': 22*60 + 28,        # 22.28 -> 1348 s
        'start_idx': (2*60 + 28) * SAMPLING_RATE,
        'end_idx': (22*60 + 28) * SAMPLING_RATE
    },
    'fun': {
        'valence': 8,
        'arousal': 4,
        'start_sec': 28*60 + 44,      # 28.44 -> 1724 s
        'end_sec': 35*60 + 26,        # 35.26 -> 2126 s
        'start_idx': (28*60 + 44) * SAMPLING_RATE,
        'end_idx': (35*60 + 26) * SAMPLING_RATE
    },
    'medi1': {
        'valence': 8,
        'arousal': 1,
        'start_sec': 40*60 + 7,       # 40.07 -> 2407 s
        'end_sec': 47*60 + 5,         # 47.05 -> 2825 s
        'start_idx': (40*60 + 7) * SAMPLING_RATE,
        'end_idx': (47*60 + 5) * SAMPLING_RATE
    },
    'tsst': {
        'valence': 4,
        'arousal': 9,
        'start_sec': 55*60 + 21,      # 55.21 -> 3321 s
        'end_sec': 66*60 + 45,        # 66.45 -> 4005 s
        'start_idx': (55*60 + 21) * SAMPLING_RATE,
        'end_idx': (66*60 + 45) * SAMPLING_RATE
    },
    'medi2': {
        'valence': 2,
        'arousal': 6,
        'start_sec': 83*60 + 28,      # 83.28 -> 5008 s
        'end_sec': 90*60 + 25,        # 90.25 -> 5425 s
        'start_idx': (83*60 + 28) * SAMPLING_RATE,
        'end_idx': (90*60 + 25) * SAMPLING_RATE
    }
}


wesad_s14_info = {
    'base': {
        'valence': 5,
        'arousal': 2,
        'start_sec': 2*60 + 0,      # 2.00 -> 120 s
        'end_sec': 22*60 + 0,       # 22.00 -> 1320 s
        'start_idx': (2*60 + 0) * SAMPLING_RATE,
        'end_idx': (22*60 + 0) * SAMPLING_RATE
    },
    'tsst': {
        'valence': 3,
        'arousal': 7,
        'start_sec': 33*60 + 18,    # 33.18 -> 1998 s
        'end_sec': 44*60 + 53,      # 44.53 -> 2693 s
        'start_idx': (33*60 + 18) * SAMPLING_RATE,
        'end_idx': (44*60 + 53) * SAMPLING_RATE
    },
    'medi2': {
        'valence': 4,
        'arousal': 3,
        'start_sec': 62*60 + 22,    # 62.22 -> 3742 s
        'end_sec': 69*60 + 19,      # 69.19 -> 4159 s
        'start_idx': (62*60 + 22) * SAMPLING_RATE,
        'end_idx': (69*60 + 19) * SAMPLING_RATE
    },
    'fun': {
        'valence': 7,
        'arousal': 2,
        'start_sec': 73*60 + 24,    # 73.24 -> 4404 s
        'end_sec': 79*60 + 56,      # 79.56 -> 4796 s
        'start_idx': (73*60 + 24) * SAMPLING_RATE,
        'end_idx': (79*60 + 56) * SAMPLING_RATE
    },
    'medi1': {
        'valence': 7,
        'arousal': 2,
        'start_sec': 84*60 + 29,    # 84.29 -> 5069 s
        'end_sec': 91*60 + 26,      # 91.26 -> 5486 s
        'start_idx': (84*60 + 29) * SAMPLING_RATE,
        'end_idx': (91*60 + 26) * SAMPLING_RATE
    }
}


wesad_s15_info = {
    'base': {
        'valence': 6,
        'arousal': 5,
        'start_sec': 4*60 + 25,   # 4.25 -> 265 s
        'end_sec': 24*60 + 20,    # 24.2 -> 1460 s
        'start_idx': (4*60 + 25) * SAMPLING_RATE,
        'end_idx': (24*60 + 20) * SAMPLING_RATE
    },
    'fun': {
        'valence': 7,
        'arousal': 5,
        'start_sec': 29*60 + 26,  # 29.26 -> 1766 s
        'end_sec': 35*60 + 58,    # 35.58 -> 2158 s
        'start_idx': (29*60 + 26) * SAMPLING_RATE,
        'end_idx': (35*60 + 58) * SAMPLING_RATE
    },
    'medi1': {
        'valence': 8,
        'arousal': 2,
        'start_sec': 40*60 + 11,  # 40.11 -> 2411 s
        'end_sec': 47*60 + 8,     # 47.08 -> 2828 s
        'start_idx': (40*60 + 11) * SAMPLING_RATE,
        'end_idx': (47*60 + 8) * SAMPLING_RATE
    },
    'tsst': {
        'valence': 4,
        'arousal': 8,
        'start_sec': 54*60 + 14,  # 54.14 -> 3254 s
        'end_sec': 66*60 + 0,     # 66 -> 3960 s
        'start_idx': (54*60 + 14) * SAMPLING_RATE,
        'end_idx': (66*60 + 0) * SAMPLING_RATE
    },
    'medi2': {
        'valence': 7,
        'arousal': 4,
        'start_sec': 80*60 + 38,  # 80.38 -> 4838 s
        'end_sec': 87*60 + 35,    # 87.35 -> 5255 s
        'start_idx': (80*60 + 38) * SAMPLING_RATE,
        'end_idx': (87*60 + 35) * SAMPLING_RATE
    }
}


wesad_s16_info = {
    'base': {
        'valence': 7,
        'arousal': 2,
        'start_sec': 3*60 + 0,       # 3.00
        'end_sec': 23*60 + 0,        # 23.00
        'start_idx': (3*60 + 0) * SAMPLING_RATE,
        'end_idx': (23*60 + 0) * SAMPLING_RATE
    },
    'tsst': {
        'valence': 3,
        'arousal': 8,
        'start_sec': 34*60 + 18,     # 34.30 -> 34m + 30s  (0.3min*60=18s)
        'end_sec': 46*60 + 3,        # 46.03 -> 46m + 3s
        'start_idx': (34*60 + 18) * SAMPLING_RATE,
        'end_idx': (46*60 + 3) * SAMPLING_RATE
    },
    'medi1': {
        'valence': 8,
        'arousal': 2,
        'start_sec': 63*60 + 9,      # 63.15 -> 63m + 15s
        'end_sec': 70*60 + 10,       # 70.10 -> 70m + 10s
        'start_idx': (63*60 + 15) * SAMPLING_RATE,
        'end_idx': (70*60 + 10) * SAMPLING_RATE
    },
    'fun': {
        'valence': 7,
        'arousal': 4,
        'start_sec': 73*60 + 22,     # 73.22 -> 73m + 22s
        'end_sec': 79*60 + 50,       # 79.50 -> 79m + 50s
        'start_idx': (73*60 + 22) * SAMPLING_RATE,
        'end_idx': (79*60 + 50) * SAMPLING_RATE
    },
    'medi2': {
        'valence': 7,
        'arousal': 1,
        'start_sec': 84*60 + 56,     # 84.56 -> 84m + 56s
        'end_sec': 91*60 + 53,       # 91.53 -> 91m + 53s
        'start_idx': (84*60 + 56) * SAMPLING_RATE,
        'end_idx': (91*60 + 53) * SAMPLING_RATE
    }
}


wesad_s17_info = {
    'base': {
        'valence': 7,
        'arousal': 2,
        'start_sec': 3*60 + 2,     # 3.02 min -> 182 s
        'end_sec': 23*60 + 3,      # 23.03 min -> 1383 s
        'start_idx': (3*60 + 2) * SAMPLING_RATE,
        'end_idx': (23*60 + 3) * SAMPLING_RATE
    },
    'fun': {
        'valence': 8,
        'arousal': 2,
        'start_sec': 30*60 + 30,   # 30.30 min -> 1830 s
        'end_sec': 37*60 + 2,      # 37.02 min -> 2222 s
        'start_idx': (30*60 + 30) * SAMPLING_RATE,
        'end_idx': (37*60 + 2) * SAMPLING_RATE
    },
    'medi1': {
        'valence': 7,
        'arousal': 2,
        'start_sec': 43*60 + 53,   # 43.53 min -> 2633 s
        'end_sec': 49*60 + 50,     # 49.50 min -> 2990 s
        'start_idx': (43*60 + 53) * SAMPLING_RATE,
        'end_idx': (49*60 + 50) * SAMPLING_RATE
    },
    'tsst': {
        'valence': 1,
        'arousal': 9,
        'start_sec': 60*60 + 2,    # 60.02 min -> 3602 s
        'end_sec': 72*60 + 25,     # 72.25 min -> 4345 s
        'start_idx': (60*60 + 2) * SAMPLING_RATE,
        'end_idx': (72*60 + 25) * SAMPLING_RATE
    },
    'medi2': {
        'valence': 5,
        'arousal': 2,
        'start_sec': 89*60 + 9,    # 89.09 min -> 5349 s
        'end_sec': 96*60 + 3,      # 96.03 min -> 5763 s
        'start_idx': (89*60 + 9) * SAMPLING_RATE,
        'end_idx': (96*60 + 3) * SAMPLING_RATE
    }
}

wesad_all_info = {
    "S2": wesad_s2_info,
    "S3": wesad_s3_info,
    "S4": wesad_s4_info,
    "S5": wesad_s5_info,
    "S6": wesad_s6_info,
    "S7": wesad_s7_info,
    "S8": wesad_s8_info,
    "S9": wesad_s9_info,
    "S10": wesad_s10_info,
    "S11": wesad_s11_info,
    "S13": wesad_s13_info,
    "S14": wesad_s14_info,
    "S15": wesad_s15_info,
    "S16": wesad_s16_info,
    "S17": wesad_s17_info,
}