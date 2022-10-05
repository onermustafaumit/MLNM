#!/bin/bash

mkdir logs

PATCH_LEVEL=3
CHECK_FIT_PATCH_LEVEL=1

echo "##### crop all patches using xml files - PATCH_LEVEL="$PATCH_LEVEL" #####"
python crop_patches_by_using_xml_check_fit.py --patch_level $PATCH_LEVEL --check_fit_patch_level $CHECK_FIT_PATCH_LEVEL --slide_list_filename slide_ids_list.txt > ./logs/log_crop_patches_by_using_xml_check_fit__patch_level3 2>&1 &
# sleep 2 # sleeps for 2sec
# wait


PATCH_LEVEL=2

echo "##### crop all patches using xml files - PATCH_LEVEL="$PATCH_LEVEL" #####"
python crop_patches_by_using_xml_check_fit.py --patch_level $PATCH_LEVEL --check_fit_patch_level $CHECK_FIT_PATCH_LEVEL --slide_list_filename slide_ids_list.txt > ./logs/log_crop_patches_by_using_xml_check_fit__patch_level2 2>&1 &
# sleep 2 # sleeps for 2sec
# wait


PATCH_LEVEL=1

echo "##### crop all patches using xml files - PATCH_LEVEL="$PATCH_LEVEL" #####"
python crop_patches_by_using_xml_check_fit.py --patch_level $PATCH_LEVEL --check_fit_patch_level $CHECK_FIT_PATCH_LEVEL --slide_list_filename slide_ids_list.txt > ./logs/log_crop_patches_by_using_xml_check_fit__patch_level1 2>&1 &
# sleep 2 # sleeps for 2sec
# wait


PATCH_LEVEL=0

echo "##### crop all patches using xml files - PATCH_LEVEL="$PATCH_LEVEL" #####"
python crop_patches_by_using_xml_check_fit.py --patch_level $PATCH_LEVEL --check_fit_patch_level $CHECK_FIT_PATCH_LEVEL --slide_list_filename slide_ids_list.txt > ./logs/log_crop_patches_by_using_xml_check_fit__patch_level0 2>&1 &
# sleep 2 # sleeps for 2sec
# wait