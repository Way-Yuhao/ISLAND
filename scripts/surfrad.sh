# define variables
surfrad_dir="/home/yuhaoliu/Data/ISLAND/surfrad_val"
start_date="20130411"
end_date="20201231"
nlcd_year="2016"

# download data
# python region_sampler.py -s BND --nlcd_year $nlcd_year --dir $surfrad_dir --start_date $start_date --end_date $end_date -r
# python region_sampler.py -s TBL --nlcd_year $nlcd_year --dir $surfrad_dir --start_date $start_date --end_date $end_date -r
# python region_sampler.py -s DRA --nlcd_year $nlcd_year --dir $surfrad_dir --start_date $start_date --end_date $end_date -r
python region_sampler.py -s FPK --nlcd_year $nlcd_year --dir $surfrad_dir --start_date $start_date --end_date $end_date -r
# python region_sampler.py -s GWN --nlcd_year $nlcd_year --dir $surfrad_dir --start_date $start_date --end_date $end_date -r
# python region_sampler.py -s PSU --nlcd_year $nlcd_year --dir $surfrad_dir --start_date $start_date --end_date $end_date -r
# python region_sampler.py -s SXF --nlcd_year $nlcd_year --dir $surfrad_dir --start_date $start_date --end_date $end_date -r
### python region_sampler.py -s SGP --nlcd_year $nlcd_year --dir $surfrad_dir --start_date $start_date --end_date $end_date

## run ISLAND
# python main_lst.py --dir $surfrad_dir/BND --skip_to_ref
## python main_lst.py --dir $surfrad_dir/TBL
# python main_lst.py --dir $surfrad_dir/DRA --skip_to_ref
#python main_lst.py --dir $surfrad_dir/FPK
#python main_lst.py --dir $surfrad_dir/GWN --skip_to_ref
#python main_lst.py --dir $surfrad_dir/PSU --skip_to_ref
#python main_lst.py --dir $surfrad_dir/SXF --skip_to_ref
#python main_lst.py --dir $surfrad_dir/SGP --skip_to_ref
