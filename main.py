import streamlit as st
from PIL import Image
from utils.tools import get_exif_tags, get_construction_site
import io

#general setting
st.set_page_config(layout="wide")
chosen_tags = [256, 257, 34853]
chosen_tags = ['ImageWidth', 'ImageLength', 'GPSInfo']

con_sites_gps = {"Uttenhofen Bach" : [47.80767, 8.63481], "Shey Schabelhof" : [47.85286, 8.63079]}
con_sites = ["Uttenhofen Bach", "Shey Schabelhof"]


#### TESTING #####
tags = get_exif_tags("assets/Uttenhofen.JPG")
get_construction_site(con_sites_gps, tags["GPS Latitude"], tags["GPS Longitude"])


#Sidebar
logo = Image.open( "assets/logo.png")
st.sidebar.image(logo)
file = st.sidebar.file_uploader("Upload Picture", type=['png', 'jpg'], accept_multiple_files=False)

# Style
st.markdown(
    """
<style>
.sidebar .sidebar-content {
    background-color: white !important;
}
</style>
""",
    unsafe_allow_html=True,
)



### IMAGE AND DETAILS
col1, col2 = st.beta_columns(2)
# col Image
col1.header("Selected Image")
if file != None:
    try:
        img_upload = Image.open(file)
        col1.image(img_upload, use_column_width=True)
        #Display exif data
        col2.header("Details")
        exif_tags = get_exif_tags(file.getvalue())
        for key, val in exif_tags.items():
            col2.write(f'{key} : {val}')

        if "GPS Latitude" in exif_tags:
            options = get_construction_site(con_sites_gps, exif_tags["GPS Latitude"], exif_tags["GPS Longitude"])
        else:
            options = con_sites

        option = st.selectbox('Construction side', options)

        ## Matching
        st.header("Match with 3D Point Cloud and Calculate Camera Pose")
        if st.button("Start"):
            st.write("Coming Soon...")



    except:
        col2.warning("No metadata found")
else:
    st.warning("No image uploaded")






