
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
    weekend = get_weekdays_by_loc(lat_long[0], lat_long[1], 'weekend') if lat_long != (-1.0, -1.0) else ['Saturday', 'Sunday']
    return df.loc[df[col_name].isin(weekend)]

def filter_by_workdays(df, lat_long, datetime_col, col_name):
    df[col_name] = pd.to_datetime((df[datetime_col]).dt.date).dt.day_name()
    workdays = get_weekdays_by_loc(lat_long[0], lat_long[1], 'workdays') if lat_long != (-1.0, -1.0) else ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    return df.loc[df[col_name].isin(workdays)]


def get_weekdays_by_loc(lat, long, flag):
    country = 'Albania'
    try:
        geolocator = Nominatim(user_agent='Louie7ai')
        location =  geolocator.reverse([lat, long], exactly_one=True, language='en')
        country = location.raw['address'].get('country')
    except Exception:
        print('geo location server not responding !!!')
    week_days = week_days_by_country(country)
    return  week_days.get(flag)


SS = ['Albania', 'Angola', 'Argentina', 'Armenia', 'Azerbaijan', 'Austria', 'Australia', 'Benin', 'Belarus', 'Belgium',
      'Brazil', 'Burundi', 'Bulgaria', 'Canada', 'Cambodia', 'Cameroon', 'Chile', 'China', 'Croatia', 'Colombia',
      'Costa Rica', 'Czech Republic', 'Denmark', 'Dominican Republic', 'Ethiopia', 'Estonia', 'Finland', 'France',
      'Gabon', 'Gambia', 'Ghana', 'Greece', 'Hungary', 'Indonesia', 'Ireland', 'Italy', 'Japan', 'Kazakhstan', 'Kenya',
      'Latvia', 'Lesotho', 'Lithuania', 'Madagascar', 'Maldives', 'Malawi', 'Mali', 'Malta', 'Mauritania', 'Malaysia',
      'Mexico', 'Mongolia', 'Morocco', 'Mozambique', 'Myanmar', 'Netherlands', 'New Zealand', 'Nigeria', 'Norway',
      'Philippines', 'Poland', 'Portugal', 'Romania', 'Russia', 'Rwanda', 'Senegal', 'Serbia', 'Singapore', 'Slovakia',
      'Spain', 'Sri Lanka', 'South Africa', 'South Korea', 'Suriname', 'Swaziland', 'Sweden', 'Switzerland', 'Seychelles',
      'Taiwan', 'Tanzania', 'Togo', 'Thailand', 'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Ukraine', 'United Kingdom',
      'United States', 'Uganda', 'Venezuela', 'Vietnam', 'Congo', 'Democratic Republic of Congo', 'Zambia', 'Zimbabwe']
FS = ['Algeria', 'Bahrain', 'Bangladesh', 'Egypt', 'Iraq', 'Israel', 'Jordan', 'Kuwait', 'Lebanon', 'Libya', 'Oman',
      'Pakistan', 'Qatar', 'Saudi Arabia', 'Sudan', 'Syria', 'United Arab Emirates', 'Yemen']
TF = ['Afghanistan']
SU = ['Bolivia', 'Equatorial Guinea', 'Germany', 'Hong Kong', 'India', 'North Korea']
ST = ['Nepal']
FR = ['Iran', 'Somalia']


def week_days_by_country(country):
    if country in SS:
        week_days = {'workdays' : ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
                     'weekend' : ['Saturday', 'Sunday']}
    elif country in FS:
        week_days = {'workdays': ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday'],
                     'weekend': ['Friday', 'Saturday']}
    elif country in TF:
        week_days = {'workdays': ['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday'],
                     'weekend': ['Thursday', 'Friday']}
    elif country in SU:
        week_days = {'workdays': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'],
                     'weekend': ['Sunday']}
    elif country in ST:
        week_days = {'workdays': ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
                     'weekend': ['Saturday']}
    elif country in FR:
        week_days = {'workdays': ['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday'],
                     'weekend': ['Friday']}
    else:
        week_days = {'workdays': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
                     'weekend': ['Saturday', 'Sunday']}
    return week_days