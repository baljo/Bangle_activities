// Activity classification using TensorFlow on Bangle.js 2
// Created on 2024-10-17 12:30:00 by Thomas Vikström

var model=atob("HAAAAFRGTDMUACAAHAAYABQAEAAMAAAACAAEABQAAAAcAAAAjAAAAOQAAAC4HQAAyB0AAAwjAAADAAAAAQAAABAAAAAAAAoAEAAMAAgABAAKAAAADAAAABwAAAA8AAAADwAAAHNlcnZpbmdfZGVmYXVsdAABAAAABAAAAJz///8KAAAABAAAAAgAAABvdXRwdXRfMAAAAAABAAAABAAAAC7i//8EAAAAAQAAAHgAAAACAAAANAAAAAQAAADc////DQAAAAQAAAATAAAAQ09OVkVSU0lPTl9NRVRBREFUQQAIAAwACAAEAAgAAAAMAAAABAAAABMAAABtaW5fcnVudGltZV92ZXJzaW9uAA4AAADQHAAAyBwAAJAcAAAwHAAAEBwAAJAEAABgAQAAsAAAAKgAAACgAAAAmAAAAJAAAABwAAAABAAAANbi//8EAAAAXAAAAAwAAAAIAA4ACAAEAAgAAAAUAAAALAAAAAAACgAMAAgAAAAHAAoAAAAAAAABBAAAAAAAAAAAAAoAEAAMAAgABAAKAAAAAwAAAAIAAAAEAAAABgAAADIuMTEuMQAAPuP//wQAAAAQAAAAMS41LjAAAAAAAAAAAAAAALje//+83v//wN7//8Te//9q4///BAAAAKAAAAD6nkQ+KrsSP+ASID9CYIS+aX5ePncCo7+vGJq/Gs4Dv0UiAT+Loy8+TioePxs16L5EWvi+8l47v496Vr/5YZ+++La6Pso7+j54fjC/og2qPn2A9b52ehc/1abZvosnVr+ecMg+SukEPsQxNz/EV+A+LrcgP0gADb8sti6+OtilvlABhryBoQ4/aMWoPsmzRL9wlBA/rmwDvypBfb+PJmS/FuT//wQAAAAgAwAA2parPgpkVb86NPi9SqdqPs4wvD6IgB4+LKr8PtgH0L49N4Q+tTgWvkz9tL2iYoY+pbxbPrfcBD9W3R4+s0+avoIrgbxY/Fy9NkthvnmUKD5e+4K+rgBevIOsKD5LWGQ+cbEMP3XzJ71QOs69WO+wvj+lCr+3J4M+51cPvbkH+b0lLe8+65NUPsV9Cj8C0RE/YYn8vF7s6D0DovS6BichPZKJ3r2AWr2+GgzuvLd7wj7KbkW+nxFsvh0sJb4obKE9LzU6vpKDib17luw9V8wtPmxH070rDam+3kJwvdV6lL6wSZ4+CNa+vh5GyL7/SZ6+Z/W9PvMtDb9GTyI/otuvvuFGo74wCug+KHPvvhtjeb5vE/O+VrkQP12yKz6pxS++SIvDvlkl5b71CPo9S6+DvkqlxL7RjFO+xQglP4X3bz5+tXm+/NJEP+INvr77iaq+qb4lvpnBpT64XRW+oaHMPVm8Jr9oROE+9qcDPwgwcD3NuYw9ZdcxPjaoyD5053u+J6ULvrxKtz45r/o+ixKWPhXjXr470H0/LwzyvtjG2T77WiM+Donovg9+jT55FZ4+ONSSvbYVQb1LGUA/QbsJPz4qFT2+Xpg+FvRcPjODQj3vL+Q9kvJNPplx770Zgcs98mVgP0kyXj8jhg29huaGPmqwgz5DPM0+oA9SPo4KGz27PPU+YR+/PC+yDD/EcwY/sDufvklRc71FkJq9jqhbvW8jyz6wRxa9S7DqvPQ37L2p2ww+pDCBP9ZR+r3oHjg/TSlPvqGKvz7Y/ys/RcWzvlo1bT/J/9S+FyBhO4RXZD+Xdbs+jXXTPdFdGL9Szqo+Djw3P8/NWD1RKVe+72QBPyCO273nOgk/MYHkPZ6KMz7vXEY/nMnsvEm0x77yYZm68ZVdvhE4tL0utwk/HaKcPWwzOj7YrKG+jVQhPo49Gj865JG+ejuJPnSttr6AoRi9+nA3vavjXr/sxnG+NtpLPqfSAz+VeOQ+iC6XPcmPbT3P+RM/k2TnvhJcH79dQNI+N/P/vRDDqb3KKLo+UT38PfRvHz0Yg8U+1gPXvXIE7j5C5///BAAAAHAXAAAwl4S+9m2kPj70+732/kG9Z5LNPnxspTwX0ya+79IOPrp1BLw4KMq+xDuQPu2L0TxOnti+Bho3PlZcAL5KEtc8It7FPudkyT2To5y+9gMcPll83j3bmpe+6aEBP5+Vvb12n2i+zP4gPuQY972QGfm9yKIvvqyjY76gCHI9cK+bPXRIGD4kiFG+Oq1uvrjo2T1Uxx2+xlw2PjASv7xg5yO+bM7iPfAcs7yl6T2+uK2ZvfCGW77iQ1A+3tKyvUBFXT6WcyW+FL01vqTMNL10hWy+gD3IPbRtXb2wh1W+PGchPnZ4bz6C0zw+EKJ5PtCWFD6Mb7894PXLvQDw1Tw04zK+nGgdvW3TMb6A+3o73JGiPaCcVT40DgO9GPJwPU1hV76kVxy+umuGvewBV70wPtA+vlhkPs7itz33+x8/TvslPmmInb0fSiY/e1J/PmO4fry3O0U/tFIJvW6wzbyboBI/g7IJPnFGGT3QxOc+tAyKPvgjjDw6rl4/erJMvQkZGb6Svxo/EByQPrmeHL4Y7gA+WKL1PVB+Nb2QVjc+hPJ0vYZHOr4AGh+9HoFsPi66Wj7cPXU+WPksPvi0n73+2Em+sMm6vEDIhD2g4uW9pBnTvQ42C76W3nY+ZOIxPo7cET5kEoW9kqs7Pt9kML5w2qM8AIQnOt2RZ77Qhum8wEAXPHogvr1QqgW93EaAvriccL66eEk+mllTPlTuSr50zTS9ZvMtPuj1zL1Mzy29UOR5vQDAJDuYqj4+7Gd8vmpICz4epYe9qP5kPjEJEL5+Kxc+MKl+vGDdET36s3a9rlvQvZk9ir0uC82+un9ZPRrr/Lut9Im9n1E3Pm6NqT4gEGm+aV1Uvgv0DT6QdMC+EMCCPTSVAr2ogsu89GTtOy8Rgz794VC+x2I6PbK2rz6GAuG+Mwi1PQsM670wefg8VsyAvRLXcj586fo9j9gCvrYzRr5mWgC+bCMnPkDzAT75Aze+mO1AvZYCpr0r1Fm+AKqAvDRrEj6QrEG+thkBPtpbAz7DVGy+iEgovdDYtT2geUC94r0DvsSGS77gC7Q9rkaDvfJTAj5cM2O9iBI2vQDmkbveOgU+/sVPvtAb2jzoIwo9sHHpPNClir2gCWk+guiuvbzcIL34dCs91uCLvaqREz4gQFI9AcAOvmTxmb3Oi6q9QNJRvCOcd76QjpE8Su1WPtgkLb31VO+9nPThPh0/Ur5042M+8PebPifhHj2ebYA91kYSP9fX3jweX7C9rKQJP1M8FT55vlg+Lkj9Pg3SYD0+qkI9ETz8PXGhKr7aAms+JqzjPmNOY74c9lE+JvQFPgdRO75Scxk+Cg4BPt7fNb6ekhY+ZOtTPjzCJD7c5OG9aLzRPZBWVD7am+G91m8ZPjAi/zw2QHg+ikZ0vmhVPz1q9S0+ebBRvpicGj4AnUg7vkVEPh61cj6gYkS+YCMsvDUVIr4i4BY+7AjqPfMJJr7ghAS8PDBIve3sf77YK3E9vlxDvpp5Wb5Q5Hm+JH/OvSBy7rzaxGQ+wGatPdrkbz5svTU+uJYovmHYZL7o7A8+O0pbvtHFIL4hqGi+QokRPkxTBb6lmwm+BOo1voBQWz1AD/Y+YlmIvb+B0DzhAQY+ra+fvit5Sr5WZAM/qzU3vhXaBL4Po+0+gC1MvhNQCT2+lh8+4MNzvrGcPb71ZuI+f4DNOtXdFj5dGZQ+pURsvih6Fr1Y/gg/tvd8vuGjML3ORqO9AIAuPBpwNz6czSs+2PVpvv6PSr5A4uU9byklvvowYT7YUze+Rlqkvd6o+r3GQZ+9ObZqvmg4Vz3sUhI+gEvFPOxBlz02PV++5vK4veRVDT6P8R6+UuE5PjhOGD5MQkm9etQqPiS/ID48wzc+RQxGvoSZg70G7oy9IIJzPpCxIT6YJUM+em9dPkScEz5ek1q+4ORMvhR6Zj4IhGk+4MtNPBRQDj6UT9y9opl6PheLer4sC3a9c+RcvmRyhj0qqi4+FOWuPeTszz1b86i+W7rhPgcIwL1AaYS+3PgqPoEqqr467Fa+41H8Pmb5Y76hFh6+MXR8PKyBV76uLu6+/ivIPk3nsr2kP2C+3PcRPegjgb1zqpC+5VqAPgWsqr492+W+uN9YPieQjr7gKjo+SoeEvdTKij2U3w+9Wr3xva4uez4M10W9QLTQvIrGRD78eRu+aPppPTQPbb72PC0+1MsTvkC/zz2Sm14+nGIuPvi8wT0coC++KjuKveB9LT23+Hm++709vmfxKb7g//i76JwhPrSmXz7EaxO9LFGoPUqmzr2wTeE9/tcXPlSRqz0AOfq85GMPPtBOkjxcM4M9B2lyvkIHJz5710e+wn+MvYC5wLresbm9cHZKPnYXXD60uuQ9xhxtPm6w5b3KoS0+vBQ7vlVGfb7PRN09+84TPkBllL1qexu+BfBfPY63gb7gz5i9imLBPmrqo728T709+4JEPo2hFT3QqAA9rWu/PvJEFj5rzTI7/izgPbEno71nX6s+gpq6PqEnwr1myok9QH5/PSscBT6cusq9aCHFvI77fj4+x1K+ZOVNPog7gj1SXCo+mD1svoBKPzuAVvu8AINGvvISFD6HKme+8E7RPdYtTb5Sxn4+2AjavDBOXD1oC568AOQ8PBB2/L10TFq+nih2PnGJCb6QOMe8vB1WvUoSKL7KilI+2LNbvt6LHD6g5KQ9wLXdvEBsrzyKUVM+OtAKPsRStr2RVx6+UBDSPFpnNT7MThq+dEPzPdR4hT1SrGS+uoshPlXHPL6rSgq+uPwoPYLnYT7SdJa94G3sPaCTlD3i0Rk+CwuXvqkdxj070E8+ejYiPnzhzT2yCb89Gh8WPWzgO70N0Hw+dpH5vTrwiz5dmiS9OhERPRkPDL4npz++esr2vTPEA76BvPG9ZD9APjXpcj5J2tS91XU9vt2lOT2cKWi9XhtzPgadjr0Ukog9g69xvkjPGL0w+Fs9jGJqPmj8VD2HuF++UiG6vee4bb58NTe9mFqJPZzeLD5sX669DBQ5vfy9jj2++H8+TPN0PrDurz2gus68cN9OPWN2WL6aLhe+RA/4PX7rLr6gnYm97GBAvWBLHD2gUqK95JUgvizxXL5cxi4+/hAnPt7fe75NYSy+8OGmvST0AD7kjzm+VrIFPqx5ij1Qx3A9+oLSvcgr7L3iIX8+8i8bvhDTKz38NUS9KKPNPUACmLu3EVA9b5cFP/lxyTyV3Oy9HAyvPXl03bsIjTG9ADA0PoKHL748nJM9dGOtPnqPTz1gPU4+KZ90PmVOU73fqGq7coe4PngrRr4bfuA+UTNvPr8l/D1qE40+OL3jPljUgr1MQB0+7EYiPk5XNj6gx7y87sdSPpJuXr72ow0+IHRxvlS2Fb2AF2O7A3BRvriHj72di16+WttWPoi9KL4wGL88tXcAvl3DA77N6Wa+VkTFvZwLX77o+pU9lPOxvSyEAL2J6wS+XJyYPdkwa75LUXm+ogF9PuQyIL6COQI+MDzzPRAOCD04+e49OBXPvOq1vr2APiI+OD1ovfCGVz7KAMu9YOdlPB6hcr5wTls+pPw2Pg6oHT50EwK+minXvRxqmr0wIig9D7dMvsAenr3sn6S+WuTBvRCEiL60yA++mBa3vuto2T1pQ8C+Ioo3vmp8Qr4YUN6+UM/Vvkiwer4IUdW+J7CqvhlhW743w0U8cxhjvu0Grj10xhq9SzR9vqoX7L2TKhC9XRWmvYsLTT5HNoA+jDwsPiDMWDxgD5c9IAHiu6AI5z3YzuC9oMqQvagwOr4AE/m7hDQzPljOmz02ryw+6AxBPijQE71gzxQ98HBmPeDT1T0Rjgq+2Gd2Pd5d2L3AZX++eNMkPqwu0T0uN1s+sr1GvqiYAz5i/HE+yRJ2vlTncL6lPRi+QME5PQ4LDz4AbNk8tlRovijdmj0Bs2O+5P8LvkotMb4630A+gCs/O1ylb718qm++/KSRvST/Lr4UgdQ9NGggvnYVKT4IGF0+gMKuO2RsN72OnRI/WADnvZBZ/j0J4Js9sG/GvlEmor08IOw+kg8yvi6atzvAB84+y8bavix9wz2y/8I++djjvncMCz6D/iE+XmuFvi1HiDz2lrk+W3FPvguXMr6jjbc+U4/4vrbrez7WmlI+xDA3vVBZ873Y2aC95kw1PngTLT5yu3S+ni9KvjCcSb4IiV29Wsp+PkSiYD6iLJG9oGZHvt3GR76UrI495GPCPfDEAr1gxE48WmOfvSgGp736pTA+0JCPPBu+cb7YwhU+PMDEPfJ+nr0gsR6+MBlIvQDSf7zkqjU++vlgPt7+Jr5I6Ni9UOthPQB/I73APIQ9LLGsPXb6Iz48NSs+kC9/PToQeL6Yc2W+WGbhPVD25j3kHI09nJFGPrAL4jxoOma9bmXQvUBJZLs2k548EPPyPuua+j1aD369/9pVPjRUqb4yF4g+B2SXPoDxuT0V1Ak9YIAdP5jDLT6Mg0M+VB8lP4LPGr6IiTg+3W+RPpcgEr4SGcg+ata5PtLM+r3dIW49QpEXP6zyrr3UE5w9bO/JPdzjID6cN/g9Y5KAPhwYxD1Y2Ku80GJXvaBYJjx8jNC9/LaIPYhcZL04KzG+QJl/PYQvhD1Sfgc+LP/ePaZDNz4kglw+FBkgvdJQBr72EZ+9kCRlPVyl6z0xPly+iB+0vNykPr4wtSs+vEebPTDpqz1gTGk+frYSPvDP/z3wUzm98HTgPMjwTL5WukE+TmUIPgCs4rsu5ou9sGI0PtasGD5M+yU+eNB8vfBvT73ELGu97t5EvmDjBDzGmA++U1UZvqYjTD5YtQC9HDCyvbdpAL4xp/u7kGusPmKEq74nFsK91G8WvXQcp76p7fU9FiAdPleiub4WeWk9MgmhPiIVYb7cAyM+G2hrPpWPYr6LL1k+DP8lPhGo9rujg3I+t2FIPalQSz24sf89QhiivbDqjjyUO/a9+nxVPrq+hr0kb9m96AVyPQ4Jd75q9GQ+DGsdPuTAHT4amU0+JRoovjH0V74J+R6+CD/3vLB5HD0rLU6+Smj5vTDZfr7Kycm9QLxqPl4cED6YPhi9DtZ9PjBCKT6SsWG+6B1/vsiTPD3+Cyi+K5p6vkPXLL7kjZo9JF5EvZA43z3GMKm9aCg4vu1PQ76qvyg+yDkjPcB8ez181Sg+FNW/PbAlrTysdC4+AKWNugjw8L2Ebuc92EkkPkRmiz2tVSc+CnGrvYrXNL7/wLw9vLzAPjROSD7MO+S9wSHIPkjMpb2hqRw+w41SPmktFz0azJw+PxWCPm2oe734YcG9gMGEvXZNN732ftO91MpiPbcrTb4+Y7q9KYQLPnGozr1Mptq9Vr5/vlhdLL4I9Fo+SLgnvWBmmDwscps9TLV3vqtcPr6nmna+8DoQvWB4dz0gq+U9QHKWvNbJTT5yJJ29nFtsPkzD+D3mUVE+Xrk4vmA8VT5QHPu9YJ9HvOT2Dz74oH4+ynYaPsDIpT08uO09Jes8vqj79L0MkWc+dnjIvfCgLj0knjA+lOj/PegXNb0AQp06+ByvvegzKr1I70S+3rwmPiApRD6KqU4+GKxIPkAGRT0ymnE+yJtUvtjuQr283CA+QMFDvQR18D0D9O4+hFUmv/R/nb7cgpQ+41fTvtUvn74lJM49VX+hvg1SW71W9SY+Ty8WvgWX3r6srse90C0tvnfu3b6RIxO9Ki8dv4+eJL79rRQ+WyYOv/oc6jxbDdQ+ZUAEv2gkxzzZgnC+duEXvijJF73Qa1K+gIFHPsDrjr0Cy38+JxALvsJI6r2YZRw+yOsfvU6rKb4VHUy+mJxmPTiXSj7YTGy+jF9/vsqETz68rHW+ciokPvzeIr3QQxs+xOy+vdz0JL5QSni+NLfZvcgXTL1Ey9895KujPeCQNr1Q+EC+BOQlvgKSZD6Eh3m9pLZbPgW1Eb6g4mw8EOLSPVh1Gr7oq2U+mLUMPb6dZD4WYFo+DJodPsRyOL0I2wU+8HvEPFrQHL4k0mQ+LHfRvYZ5Wz54MjY+gJ6iPs8Gor7UAJK9InKNvbjgbL1WRP09im9DPuIaGb24WDc+wPH6Pfu7Gb2K4j69zqVWPk3Qarz8PYg+56cePiET2rzCdzQ+A6aiPpahtr50Msk+Q0/HveuMjTtAoGk9AE7jOlpR473ee7m98DhEvYQWDb7Yk4U94n+xvZgst700pAU+oJWbvChtWT0ArMO5yQtsvmBpgT0Kw2E+ao0SPpRW870MWZY9SLs2vio+Mz74fx2+9lZlvmiSf70KOTq+EHAvPjzH0z2Igau8RK4zPjTTWD6Aics9onkRvjDvXT6smt89w5Q5vsRLVT4A9c87Pq8evnhSPT14PN296O4hPTxQ0D3YGh098KwXviC5szyK9CM+jeRbvjBXUD4Eg00+/DxcPuyx0T0Phx49sqFEPg5xrb1JPhO7rOMLP9OvS71g64e9ecwpPh6g7z0FBj2+XjkXPgwj7LyAoo09p/sAP4CyBL2tt9Q9xFQvPsg5k70YdA0+c/WAPtWyLr7QHA2+KwTSPpnJq71Q7tM9fimovaR1Nb2CPkM+Hg4zvsD1Nz7Yt7g9wLsFuyZURj5aMRm+/D6ZvbscB75o5ZY9VhBbvghF5L0AoA88APKEuhAZLL0gqS49ksM5Phgx5rypi0y+CJJGPu45ar4Y0Zq8zKjKvXDGnDxEgwG9qG88vSBoO75MJQ+9kvRyvlj9Fr1cvFO9VC0lPmauMr78OCW+FN5bPqeQFL5AWHM8FAs0vshK6D1ANYc7Jlq/vWCwr7vCW3I+4KRSviRvUL04EAS+QJZbPoiuPj6UVPe80rPcPXKScj4EJ0Y+VpwqPjatgb3vrKO99fZGPs0syj3s6328VQHQvS6+gT7/G5O9NU4nPt5NLr75AYq9vx9AvnlQkT14/ns+hGwZvlOdFT57JLi9Xj5LvsxbhD73hwm+ASFGvocRAb5076Q9oMaRvKLTE77cfQa9USZBvoCbDDwoinw9cEa2PBhRez5Q6nI+eMMIvr4gMz5gs1a+ZQ1yvjDilL2LK4A+MMebvbCSmD3dVSi+4FS2PBCXcj7XckG+xC34PSwA7b1QThO+BoBrPmCrBj4IuJM93+o9vl6JPj6KMF++bB9gvZxi872wg5k9uyBHvqxs9j2UaXA+5GZSvRIQBz6gBuO8wJS7O7BQpT3UPI+9Oo9DPgjTur2WtyI+cmBoPtjm7T1O5bm+WqKMvph3mD2zFM6+/JzVvZWagrxUHK2+UR+Fvhfucb03+JK+/4/xvrCWN763Js6+xeHfvgKxhr5GjKu+0T48vqtear4W5L2+WF8ivUPEV76Gp4G8dy2SvmztAT7g0N29Wt1evuB7IbwAqJG9pTxkvkXwML5kEOw9MoXdvcwsIr69Nma+oD18PkTtST5ox409wigsvu81Rb5MRKu9TuU5vlDkTT3uSxm+yroDPgAL2jvkYh+92VaAPpD6Zz3yIEq+UFFrvuDfYDxpLFq+GNXkPei+Er7oMiI+B6A0vtqYJr6Y+CC+aumLvTzxNj5ezaC9uskFvqycKL7YT0g+gE38PbZOBb4C4F4+EEpPvqByUzzYqGA+Iq8ePtJFCz50tEA++kDpvWApcb6tMB87a9urPo+tD740ZLi9y36qPWYnc74mFQM+FqfOPdE7yzw/Y4U+Uhd+ve2tQz3KGXE+JEA6PjdcI75sC4E7wn8hPhrCur69d4k9AtUpPrBbf77j6t+8lbaaPuNdub6YaIi8APzWO/p/Jb7WUNe9xIC0PTx/Sj67ZDK+mHgWPXQIWD4nih++QHawPHQnsT13fWm+OCpdPcB0eTxE1AK9F4YnvkYqMb4ogYY90O4OPgBlyjqUrk6+ibhuviz+iD2QZsS8IKoTPJirgj1cIes9kAJYvgxVrj2gwuw9GOgCvvBYBz5jLVW+7rMrvio/VT6c5EW+6g8rPi1yO778frQ9DLRAvkBA5LzMyiS9nFstvnLFOL7Wihe+Ltl0vl5DKT6obJa9NOQOvQS3Xb2+/v//BAAAABAAAADm0gu70XnePHB3R72xar082v7//wQAAABQAAAAFRpAPvos/z0o1wE+ASNLOysRKj3tqUY+cc0lvFjJK71ISKU9zfcsPjCOtD16bKg9KYrUPfilLb0L9iw+Y4sBPhA3W72sKoG8QfojPi/rFj42////BAAAACgAAABqL3g8rwEpPpjlvLzi6gY+FGj/PaR2eD0yWE09Cbi9vGTCaz3SYYQ9yPr//8z6//8PAAAATUxJUiBDb252ZXJ0ZWQuAAEAAAAUAAAAAAAOABgAFAAQAAwACAAEAA4AAAAUAAAAHAAAACQBAAAoAQAALAEAAAQAAABtYWluAAAAAAQAAADMAAAAhAAAAFAAAAAUAAAAAAAOABoAFAAQAAwACwAEAA4AAAAcAAAAAAAACRwAAAAgAAAAAQAAAAAABgAIAAQABgAAAAAAgD8BAAAACgAAAAEAAAAJAAAAmv///xAAAAAAAAAIDAAAABAAAACQ+///AQAAAAkAAAADAAAACAAAAAYAAAADAAAAyv///xAAAAAAAAAIEAAAABQAAAC6////AAAAAQEAAAAIAAAAAwAAAAcAAAAFAAAAAQAAAAAADgAWAAAAEAAMAAsABAAOAAAAGAAAAAAAAAgYAAAAHAAAAAAABgAIAAcABgAAAAAAAAEBAAAABwAAAAMAAAAAAAAABAAAAAIAAAABAAAACgAAAAEAAAAAAAAACwAAAKgDAAA4AwAA5AIAAIwCAABEAgAA+AEAAKwBAAA0AQAAtAAAAFAAAAAEAAAAmvz//wAAAAEQAAAAEAAAAAsAAAAoAAAAhPz//xkAAABTdGF0ZWZ1bFBhcnRpdGlvbmVkQ2FsbDowAAAAAgAAAAEAAAAEAAAA4vz//wAAAAEQAAAAEAAAAAoAAABAAAAAzPz//zIAAABzZXF1ZW50aWFsL3lfcHJlZC9NYXRNdWw7c2VxdWVudGlhbC95X3ByZWQvQmlhc0FkZAAAAgAAAAEAAAAEAAAAQv3//wAAAAEQAAAAEAAAAAkAAABcAAAALP3//0wAAABzZXF1ZW50aWFsL2RlbnNlXzEvTWF0TXVsO3NlcXVlbnRpYWwvZGVuc2VfMS9SZWx1O3NlcXVlbnRpYWwvZGVuc2VfMS9CaWFzQWRkAAAAAAIAAAABAAAACgAAAL79//8AAAABEAAAABAAAAAIAAAAVAAAAKj9//9GAAAAc2VxdWVudGlhbC9kZW5zZS9NYXRNdWw7c2VxdWVudGlhbC9kZW5zZS9SZWx1O3NlcXVlbnRpYWwvZGVuc2UvQmlhc0FkZAAAAgAAAAEAAAAUAAAAMv7//wAAAAEQAAAAEAAAAAcAAAAoAAAAHP7//xgAAABzZXF1ZW50aWFsL3lfcHJlZC9NYXRNdWwAAAAAAgAAAAQAAAAKAAAAev7//wAAAAEQAAAAEAAAAAYAAAAoAAAAZP7//xkAAABzZXF1ZW50aWFsL2RlbnNlXzEvTWF0TXVsAAAAAgAAAAoAAAAUAAAAwv7//wAAAAEQAAAAEAAAAAUAAAAkAAAArP7//xcAAABzZXF1ZW50aWFsL2RlbnNlL01hdE11bAACAAAAFAAAAEsAAAAG////AAAAARAAAAAQAAAABAAAADgAAADw/v//KAAAAHNlcXVlbnRpYWwveV9wcmVkL0JpYXNBZGQvUmVhZFZhcmlhYmxlT3AAAAAAAQAAAAQAAABa////AAAAARAAAAAQAAAAAwAAADQAAABE////JwAAAHNlcXVlbnRpYWwvZGVuc2UvQmlhc0FkZC9SZWFkVmFyaWFibGVPcAABAAAAFAAAAKr///8AAAABEAAAABAAAAACAAAAOAAAAJT///8pAAAAc2VxdWVudGlhbC9kZW5zZV8xL0JpYXNBZGQvUmVhZFZhcmlhYmxlT3AAAAABAAAACgAAAAAAFgAYABQAAAAQAAwACAAAAAAAAAAHABYAAAAAAAABFAAAABQAAAABAAAAJAAAAAQABAAEAAAAEwAAAHNlcnZpbmdfZGVmYXVsdF94OjAAAgAAAAEAAABLAAAAAgAAACAAAAAEAAAA9P///xkAAAAAAAAZDAAMAAsAAAAAAAQADAAAAAkAAAAAAAAJ");

var activity = "";  // Variable to hold the current activity label
var recording = false;
var accelBuffer = [];  // Buffer to collect accelerometer data for inference
var bufferSize = 75;  // Adjustable buffer size (x, y, z for 24 samples)
var model;  // Placeholder for the TensorFlow Lite model

// Load the TensorFlow Lite model (model data must be provided by the user)
function loadModel() {
  try {
    model = require("tensorflow").create(model.length, model);
    console.log("Model loaded successfully");
  } catch (e) {
    console.log("Error loading model:", e);
  }
}

// Function to start/stop recording using the physical button (short press)
function toggleRecording() {
  recording = !recording;
  if (recording) {
    console.log("Started recording accelerometer data for classification.");
    accelBuffer = [];  // Clear any existing buffer
    Bangle.on('accel', onAccel);
    showStatus("Recording...");
  } else {
    console.log("Stopped recording.");
    Bangle.removeListener('accel', onAccel);
    showStatus("Stopped recording");
  }
}

// Function to handle accelerometer data and perform classification
function onAccel(a) {
  // Add accelerometer data to buffer
  accelBuffer.push(a.x, a.y, a.z);

  // If buffer is full, perform inference
  if (accelBuffer.length >= bufferSize) {
    performInference();
    accelBuffer = [];  // Clear buffer after inference
  }
}



// Perform real-time activity classification based on buffer
function performInference() {
  if (!model) {
    return;
  }

  try {
    // Set the input data into the model
    model.getInput().set(new Float32Array(accelBuffer));
    
    // Invoke the model to perform inference
    model.invoke();
    
    // Get the output from the model
    let output = model.getOutput();

    // Find the activity index manually to avoid using Math.max(...)
    let maxIndex = 0;
    for (let i = 1; i < output.length; i++) {
      if (output[i] > output[maxIndex]) {
        maxIndex = i;
      }
    }
    
    let detectedActivity = getActivityLabel(maxIndex);

    // Show the detected activity on the watch screen
    showStatus("Detected: " + detectedActivity);
    console.log(detectedActivity);
    
  } catch (e) {
    // Log a simple error message without complicated characters
    console.log("Error during inference.");
  }
}

// Utility to map activity index to label (depends on the trained model)
function getActivityLabel(index) {
  const labels = ["idling", "running", "sitting", "walking"];
  return labels[index] || "unknown";
}




// Function to show status on the watch screen
function showStatus(message) {
  g.clear();
  g.setFont("6x8", 2);
  g.setFontAlign(0, 0); // Center the text
  g.drawString(message, g.getWidth() / 2, g.getHeight() / 2);
  g.flip();
}




// Main menu for the app
var mainMenu = {
  "" : { "title" : "Activity Classifier" },
  "Start/Stop Recording": toggleRecording,
  "Exit": () => { load(); }
};

// Set up the app to show the main menu
E.showMenu(mainMenu);

// Load the TensorFlow Lite model on startup
loadModel();

// Use the single button to start or stop recording (short press only)
setWatch(toggleRecording, BTN, {repeat: true, edge: "falling", debounce: 50});

// Display initial status message
showStatus("Ready to Detect Activity");
