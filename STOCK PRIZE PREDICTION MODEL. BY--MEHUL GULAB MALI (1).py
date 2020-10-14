{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: quandl in c:\\users\\admin\\anaconda3\\lib\\site-packages (3.5.2)Note: you may need to restart the kernel to use updated packages.\n",
      "Requirement already satisfied: pandas>=0.14 in c:\\users\\admin\\anaconda3\\lib\\site-packages (from quandl) (1.0.5)\n",
      "Requirement already satisfied: python-dateutil in c:\\users\\admin\\anaconda3\\lib\\site-packages (from quandl) (2.8.1)\n",
      "Requirement already satisfied: inflection>=0.3.1 in c:\\users\\admin\\anaconda3\\lib\\site-packages (from quandl) (0.5.1)\n",
      "Requirement already satisfied: six in c:\\users\\admin\\anaconda3\\lib\\site-packages (from quandl) (1.15.0)\n",
      "Requirement already satisfied: more-itertools in c:\\users\\admin\\anaconda3\\lib\\site-packages (from quandl) (8.4.0)\n",
      "Requirement already satisfied: requests>=2.7.0 in c:\\users\\admin\\anaconda3\\lib\\site-packages (from quandl) (2.24.0)\n",
      "Requirement already satisfied: numpy>=1.8 in c:\\users\\admin\\anaconda3\\lib\\site-packages (from quandl) (1.18.5)\n",
      "Requirement already satisfied: pytz>=2017.2 in c:\\users\\admin\\anaconda3\\lib\\site-packages (from pandas>=0.14->quandl) (2020.1)\n",
      "Requirement already satisfied: idna<3,>=2.5 in c:\\users\\admin\\anaconda3\\lib\\site-packages (from requests>=2.7.0->quandl) (2.10)\n",
      "Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in c:\\users\\admin\\anaconda3\\lib\\site-packages (from requests>=2.7.0->quandl) (1.25.9)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\users\\admin\\anaconda3\\lib\\site-packages (from requests>=2.7.0->quandl) (2020.6.20)\n",
      "Requirement already satisfied: chardet<4,>=3.0.2 in c:\\users\\admin\\anaconda3\\lib\\site-packages (from requests>=2.7.0->quandl) (3.0.4)\n",
      "\n"
     ]
    }
   ],
   "source": [
    "pip install quandl"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [],
   "source": [
    "import quandl\n",
    "\n",
    "import numpy as np\n",
    "from sklearn.linear_model import  LinearRegression\n",
    "from sklearn.svm import SVR\n",
    "from sklearn.model_selection import train_test_split\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "             Open   High    Low  Close     Volume  Ex-Dividend  Split Ratio  \\\n",
      "Date                                                                          \n",
      "1997-05-16  22.38  23.75  20.50  20.75  1225000.0          0.0          1.0   \n",
      "1997-05-19  20.50  21.25  19.50  20.50   508900.0          0.0          1.0   \n",
      "1997-05-20  20.75  21.00  19.63  19.63   455600.0          0.0          1.0   \n",
      "1997-05-21  19.25  19.75  16.50  17.13  1571100.0          0.0          1.0   \n",
      "1997-05-22  17.25  17.38  15.75  16.75   981400.0          0.0          1.0   \n",
      "\n",
      "            Adj. Open  Adj. High  Adj. Low  Adj. Close  Adj. Volume  \n",
      "Date                                                                 \n",
      "1997-05-16   1.865000   1.979167  1.708333    1.729167   14700000.0  \n",
      "1997-05-19   1.708333   1.770833  1.625000    1.708333    6106800.0  \n",
      "1997-05-20   1.729167   1.750000  1.635833    1.635833    5467200.0  \n",
      "1997-05-21   1.604167   1.645833  1.375000    1.427500   18853200.0  \n",
      "1997-05-22   1.437500   1.448333  1.312500    1.395833   11776800.0  \n"
     ]
    }
   ],
   "source": [
    "import quandl\n",
    "df= quandl.get(\"WIKI/AMZN\")\n",
    "print(df.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "            Adj. Close\n",
      "Date                  \n",
      "1997-05-16    1.729167\n",
      "1997-05-19    1.708333\n",
      "1997-05-20    1.635833\n",
      "1997-05-21    1.427500\n",
      "1997-05-22    1.395833\n"
     ]
    }
   ],
   "source": [
    "df = df[['Adj. Close']]\n",
    "print(df.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "            Adj. Close  Prediction\n",
      "Date                              \n",
      "2018-03-21     1581.86         NaN\n",
      "2018-03-22     1544.10         NaN\n",
      "2018-03-23     1495.56         NaN\n",
      "2018-03-26     1555.86         NaN\n",
      "2018-03-27     1497.05         NaN\n"
     ]
    }
   ],
   "source": [
    "forecast_out= 30\n",
    "df['Prediction'] = df[['Adj. Close']].shift(-forecast_out)\n",
    "\n",
    "print(df.tail())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[   1.72916667]\n",
      " [   1.70833333]\n",
      " [   1.63583333]\n",
      " ...\n",
      " [1350.47      ]\n",
      " [1338.99      ]\n",
      " [1386.23      ]]\n"
     ]
    }
   ],
   "source": [
    "X = np.array(df.drop(['Prediction'],1))\n",
    "X = X[:-forecast_out]\n",
    "print(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1.54166667e+00 1.51583333e+00 1.58833333e+00 ... 1.49556000e+03\n",
      " 1.55586000e+03 1.49705000e+03]\n"
     ]
    }
   ],
   "source": [
    "y = np.array(df['Prediction'])\n",
    "y = y[:-forecast_out]\n",
    "print(y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "SVR(C=1000.0, gamma=0.1)"
      ]
     },
     "execution_count": 64,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)\n",
    "svr_rbf.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "svm confidence:  0.9481372621078293\n"
     ]
    }
   ],
   "source": [
    "svm_confidence =svr_rbf.score(X_test, y_test)\n",
    "print(\"svm confidence: \", svm_confidence)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LinearRegression()"
      ]
     },
     "execution_count": 66,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "lr = LinearRegression()\n",
    "lr.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "lr confidence:  0.9481372621078293\n"
     ]
    }
   ],
   "source": [
    "lr_confidence =lr.score(X_test, y_test)\n",
    "print(\"lr confidence: \", svm_confidence)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1414.51]\n",
      " [1451.05]\n",
      " [1461.76]\n",
      " [1448.69]\n",
      " [1468.35]\n",
      " [1482.92]\n",
      " [1484.76]\n",
      " [1500.  ]\n",
      " [1521.95]\n",
      " [1511.98]\n",
      " [1512.45]\n",
      " [1493.45]\n",
      " [1500.25]\n",
      " [1523.61]\n",
      " [1537.64]\n",
      " [1545.  ]\n",
      " [1551.86]\n",
      " [1578.89]\n",
      " [1598.39]\n",
      " [1588.18]\n",
      " [1591.  ]\n",
      " [1582.32]\n",
      " [1571.68]\n",
      " [1544.93]\n",
      " [1586.51]\n",
      " [1581.86]\n",
      " [1544.1 ]\n",
      " [1495.56]\n",
      " [1555.86]\n",
      " [1497.05]]\n"
     ]
    }
   ],
   "source": [
    "X_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]\n",
    "print(X_forecast)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1496.03002691 1534.75691362 1546.10789766 1532.25566971 1553.09230334\n",
      " 1568.53430496 1570.48442734 1586.63652787 1609.90021598 1599.33352029\n",
      " 1599.83164937 1579.69451616 1586.90149015 1611.65956551 1626.52924861\n",
      " 1634.32973811 1641.60030305 1670.24802467 1690.91508244 1680.09402296\n",
      " 1683.08279747 1673.88330713 1662.60651254 1634.25554867 1678.32407493\n",
      " 1673.39577654 1633.37587391 1581.9307978  1645.83969951 1583.50997298]\n",
      "[1008.70002719 1550.75681182  674.14161546 1079.46637627  674.10858206\n",
      "  674.10858206  674.10858206  674.10858206  674.10858206  674.10858206\n",
      "  674.10858206  674.10858206  674.10858206  674.10858206  674.10858206\n",
      "  674.10858206  674.10858206  674.10858206  674.10858206  674.10858206\n",
      "  674.10858206  674.10858206  674.10858206  674.10858206  674.10858206\n",
      "  674.10858206  674.10858206  674.10858206  674.10858206  674.10858206]\n"
     ]
    }
   ],
   "source": [
    "lr_prediction = lr.predict(X_forecast)\n",
    "print(lr_prediction)\n",
    "\n",
    "svm_prediction = svr_rbf.predict(X_forecast)\n",
    "print(svm_prediction)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
