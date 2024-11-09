




for i in {1..156}; do echo "select ca${i}, resi $i and name CA " ; done | tr '\n' ';'



p=0 ; while read -r r1 r2; do p=$(expr $p + 1) ;  echo "dist d${p}, ca${r1}, ca${r2} " ; done < all_contact.txt | tr '\n' ';'

set dash_radius, 0.1 ; hide labels ; set dash_gap, 0 ; set dash_color, teal




p=0 ; while read -r r1 r2 v; do p=$(expr $p + 1) ;  echo "dist d${p}, ca${r1}, ca${r2} ; set dash_radius, ${v}, d${p} " ; done < permute_hl2.txt | tr '\n' ';'

hide labels ; set dash_gap, 0 ; set dash_color, teal




p=0 ; while read -r r1 r2 v; do p=$(expr $p + 1) ;  echo "dist d${p}, ca${r1}, ca${r2} ; set dash_radius, ${v}, d${p} " ; done < random_hl2.txt | tr '\n' ';'

hide labels ; set dash_gap, 0 ; set dash_color, teal
