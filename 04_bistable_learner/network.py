import nest


N_i = 500
N_e = 8 * N_i

pop_e = nest.Create('iaf_psc_alpha', N_e)
pop_i1 = nest.Create('iaf_psc_alpha', N_i)
pop_i2 = nest.Create('iaf_psc_alpha', N_i)

