import earthacccess



# Set up the Earthdata login credentials
EARTHDATA_USERNAME = 'username'
EARTHDATA_PASSWORD = 'password'

# Initialize the Earthdata login
earthaccess.login()


# Sample search for the ATL06 dataset
results = earthaccess.search_data(
    short_name='ATL06',
    bounding_box=(-10, 20, 10, 50),
    temporal=("1999-02", "2019-03"),
    count=10
)

files = earthaccess.download(results, "/local/path/")
