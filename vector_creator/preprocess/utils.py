
import pandas as pd
from geopy.geocoders import Nominatim


def calc_number_of_days(df, datetime_col):
    l = len(df[datetime_col])
    x = pd.Timestamp(df[datetime_col][0])
    y = pd.Timestamp(df[datetime_col][l-1])
    z = y - x
    return z.to_pytimedelta().days


def filter_day_hours(df, datetime_col, start_time, end_time):
    df1 = df.set_index(datetime_col)
    return df1.between_time(start_time, end_time)


# todo : Delete this function
def createColumnDayDate(df, datetime_col, col_name):
    df[col_name] = (df[datetime_col]).dt.date
    return df


def filter_by_weekends(df, lat_long, datetime_col, col_name):
    df[col_name] = pd.to_datetime((df[datetime_col]).dt.date).dt.day_name()
    weekend = get_weekdays_by_loc(lat_long[0], lat_long[1]) if lat_long != (-1.0, -1.0) else ['Saturday', 'Sunday']
    return df.loc[df[col_name].isin(weekend)]


def get_weekdays_by_loc(lat, long):
    geolocator = Nominatim(user_agent='Louie7ai')
    location =  geolocator.reverse([lat, long], exactly_one=True, language='en')
    country = location.raw['address'].get('country')
    return  week_days_by_country.get(country)


week_days_by_country = {'Afghanistan':  ['Thursday', 'Friday'],
                        'Albania':  ['Saturday', 'Sunday'],
                        'Algeria': ['Friday', 'Saturday'],
                        'Angola':  ['Saturday', 'Sunday'],
                        'Argentina': ['Saturday', 'Sunday'],
                        'Armenia': ['Saturday', 'Sunday'],
                        'Azerbaijan': ['Saturday', 'Sunday'],
                        'Austria': ['Saturday', 'Sunday'],
                        'Australia': ['Saturday', 'Sunday'],
                        'Bahrain': ['Friday', 'Saturday'],
                        'Bangladesh': ['Friday', 'Saturday'],
                        'Benin':  ['Saturday', 'Sunday'],
                        'Belarus': ['Saturday', 'Sunday'],
                        'Belgium': ['Saturday', 'Sunday'],
                        'Bolivia': ['Sunday'],
                        'Brazil': ['Saturday', 'Sunday'],
                        'Burundi': ['Saturday', 'Sunday'],
                        'Bulgaria': ['Saturday', 'Sunday'],
                        'Canada': ['Saturday', 'Sunday'],
                        'Cambodia': ['Saturday', 'Sunday'],
                        'Cameroon': ['Saturday', 'Sunday'],
                        'Chile': ['Saturday', 'Sunday'],
                        'China': ['Saturday', 'Sunday'],
                        'Croatia': ['Saturday', 'Sunday'],
                        'Colombia': ['Saturday', 'Sunday'],
                        'Costa Rica': ['Saturday', 'Sunday'],
                        'Czech Republic': ['Saturday', 'Sunday'],
                        'Denmark': ['Saturday', 'Sunday'],
                        'Dominican Republic': 	['Saturday', 'Sunday'],
                        'Egypt': ['Friday', 'Saturday'],
                        'Ethiopia': ['Saturday', 'Sunday'],
                        'Estonia': ['Saturday', 'Sunday'],
                        'Equatorial Guinea': ['Sunday'],
                        'Finland': ['Saturday', 'Sunday'],
                        'France': ['Saturday', 'Sunday'],
                        'Gabon': ['Saturday', 'Sunday'],
                        'Gambia': ['Saturday', 'Sunday'],
                        'Germany':  ['Sunday'],
                        'Ghana': ['Saturday', 'Sunday'],
                        'Greece': ['Saturday', 'Sunday'],
                        'Hungary': ['Saturday', 'Sunday'],
                        'Hong Kong': ['Sunday'],
                        'India': ['Sunday'],
                        'Indonesia': ['Saturday', 'Sunday'],
                        'Iran': ['Friday'],
                        'Iraq': ['Friday', 'Saturday'],
                        'Ireland': ['Saturday', 'Sunday'],
                        'Israel': ['Friday', 'Saturday'],
                        'Italy': ['Saturday', 'Sunday'],
                        'Japan': ['Saturday', 'Sunday'],
                        'Jordan': ['Friday', 'Saturday'],
                        'Kazakhstan': ['Saturday', 'Sunday'],
                        'Kuwait': ['Friday', 'Saturday'],
                        'Kenya': ['Saturday', 'Sunday'],
                        'Latvia': ['Saturday', 'Sunday'],
                        'Lebanon': ['Saturday', 'Sunday'],
                        'Lesotho': ['Saturday', 'Sunday'],
                        'Libya': ['Friday', 'Saturday'],
                        'Lithuania': ['Saturday', 'Sunday'],
                        'Madagascar': ['Saturday', 'Sunday'],
                        'Maldives': ['Saturday', 'Sunday'],
                        'Malawi': ['Saturday', 'Sunday'],
                        'Mali': ['Saturday', 'Sunday'],
                        'Malta': ['Saturday', 'Sunday'],
                        'Mauritania' : ['Saturday', 'Sunday'],
                        'Malaysia': ['Saturday', 'Sunday'],
                        'Mexico': ['Saturday', 'Sunday'],
                        'Mongolia': ['Saturday', 'Sunday'],
                        'Morocco': ['Saturday', 'Sunday'],
                        'Mozambique': ['Saturday', 'Sunday'],
                        'Myanmar': ['Saturday', 'Sunday'],
                        'Nepal': ['Saturday'],
                        'Netherlands': ['Saturday', 'Sunday'],
                        'New Zealand': ['Saturday', 'Sunday'],
                        'Nigeria': ['Saturday', 'Sunday'],
                        'North Korea': ['Sunday'],
                        'Norway': ['Saturday', 'Sunday'],
                        'Oman': ['Friday', 'Saturday'],
                        'Pakistan': ['Saturday', 'Sunday'],
                        'Philippines': ['Saturday', 'Sunday'],
                        'Poland': ['Saturday', 'Sunday'],
                        'Portugal': ['Saturday', 'Sunday'],
                        'Qatar': ['Friday', 'Saturday'],
                        'Romania': ['Saturday', 'Sunday'],
                        'Russia': ['Saturday', 'Sunday'],
                        'Rwanda': ['Saturday', 'Sunday'],
                        'Saudi Arabia': ['Friday', 'Saturday'],
                        'Senegal': ['Saturday', 'Sunday'],
                        'Serbia': ['Saturday', 'Sunday'],
                        'Singapore': ['Saturday', 'Sunday'],
                        'Slovakia': ['Saturday', 'Sunday'],
                        'Spain': ['Saturday', 'Sunday'],
                        'Sri Lanka': ['Saturday', 'Sunday'],
                        'South Africa': ['Saturday', 'Sunday'],
                        'South Korea': ['Saturday', 'Sunday'],
                        'Somalia': ['Friday'],
                        'Sudan': ['Friday', 'Saturday'],
                        'Suriname': ['Saturday', 'Sunday'],
                        'Swaziland': ['Saturday', 'Sunday'],
                        'Sweden': ['Saturday', 'Sunday'],
                        'Switzerland': ['Saturday', 'Sunday'],
                        'Syria': ['Friday', 'Saturday'],
                        'Seychelles': ['Saturday', 'Sunday'],
                        'Taiwan': ['Saturday', 'Sunday'],
                        'Tanzania': ['Saturday', 'Sunday'],
                        'Togo': ['Saturday', 'Sunday'],
                        'Thailand': ['Saturday', 'Sunday'],
                        'Trinidad and Tobago': ['Saturday', 'Sunday'],
                        'Tunisia': ['Saturday', 'Sunday'],
                        'Turkey': ['Saturday', 'Sunday'],
                        'Ukraine': ['Saturday', 'Sunday'],
                        'United Arab Emirates': ['Friday', 'Saturday'],
                        'United Kingdom': ['Saturday', 'Sunday'],
                        'United States': ['Saturday', 'Sunday'],
                        'Uganda': ['Saturday', 'Sunday'],
                        'Venezuela': ['Saturday', 'Sunday'],
                        'Vietnam': ['Saturday', 'Sunday'],
                        'Yemen' : ['Friday', 'Saturday'],
                        'Congo': ['Saturday', 'Sunday'],
                        'Democratic Republic of Congo': ['Saturday', 'Sunday'],
                        'Zambia': ['Saturday', 'Sunday'],
                        'Zimbabwe': ['Saturday', 'Sunday']
                     }