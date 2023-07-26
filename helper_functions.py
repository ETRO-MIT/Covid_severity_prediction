

def scale_values(value, min_value, max_value):
    return (2 * (value - (min_value)) / (max_value - (min_value))) - 1


def age_to_categorical(age):
    if age < 40:
        return -1
    elif 40 <= age < 50:
        return -0.5
    elif 50 <= age < 60:
        return 0
    elif 60 <= age < 70:
        return 0.5
    elif 70 <= age < 80:
        return 1
    else:
        return 1.5


def gender_to_categorical(gender):
    if gender == 'F':
        return 0
    elif gender == 'M':
        return 1
    else:
        print('\tgender not defined as M / F')
        return None
