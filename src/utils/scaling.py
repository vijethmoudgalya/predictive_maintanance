from sklearn.preprocessing import StandardScaler


def scale(x_train,x_test):
    scalar = StandardScaler()
    x_train_scaled = scalar.fit_transform(x_train)
    x_test_scaled = scalar.fit_transform(x_test)
    return x_train_scaled,x_test_scaled
