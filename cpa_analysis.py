import pandas as pd
import statsmodels.api as sm


def data_formatting():
    data = pd.read_csv('cpa_data.csv', header=0, encoding='Shift-JIS')
    data.columns = ['Time', 'Sex', 'Age',
                    'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10',
                    'Y1_fb', 'Y2_fb', 'Y3_fb', 'Y4_fb', 'Y5_fb', 'Y6_fb', 'Y7_fb',
                    'Y1_tw', 'Y2_tw', 'Y3_tw', 'Y4_tw', 'Y5_tw', 'Y6_tw', 'Y7_tw',
                    'Y1_in', 'Y2_in', 'Y3_in', 'Y4_in', 'Y5_in', 'Y6_in', 'Y7_in',
                    'use_fb', 'use_tw', 'use_in', 'like_fb', 'like_tw', 'like_in']

    data[['use_fb', 'use_tw', 'use_in']] = data[['use_fb', 'use_tw', 'use_in']].applymap(lambda x: convert_01(x))

    data = data.drop(['Time', 'Age', 'Sex'], axis=1)
    data = data.astype('int')

    return data


def convert_01(x):
    if x == '使用している':
        return 1
    else:
        return 0


def main():
    data = data_formatting()
    for i in ['X2', 'X9', 'X6', 'X8', 'X10']:
        data[i] = 6 - data[i]

    big5questions = {
        "E": ['X1', 'X6'],
        "A": ['X2', 'X7'],
        "C": ['X3', 'X8'],
        "N": ['X4', 'X9'],
        "O": ['X5', 'X10']
    }

    for personality, two_questions in big5questions.items():
        data[personality] = data[two_questions[0]] + data[two_questions[1]]

    modelnames = [
        "like_fb_X_big5_ols_model",
        "like_tw_X_big5_ols_model",
        "like_in_X_big5_ols_model"
    ]

    big5 = data[['E', 'A', 'C', 'N', 'O']]
    # big5 = sm.add_constant(big5)  # これの有無は平均の有り無しに影響

    for modelname in modelnames:

        model = sm.OLS(data[modelname[:7]], big5)
        print(modelname)
        results = model.fit()
        print(pd.DataFrame([results.params, results.pvalues], index=['Para', 'pvalue']))
        print('R2: {0:.3f}'.format(results.rsquared))
        print('\n--------------------\n')

    sns_characteristics = ["おしゃれさ", "手軽さ", "開放性", "リアルタイム性", "閉鎖性", "匿名性", "バーチャル性"]

    df_ave = data[['Y1_fb', 'Y2_fb', 'Y3_fb', 'Y4_fb', 'Y5_fb', 'Y6_fb', 'Y7_fb',
                'Y1_tw', 'Y2_tw', 'Y3_tw', 'Y4_tw', 'Y5_tw', 'Y6_tw', 'Y7_tw',
                'Y1_in', 'Y2_in', 'Y3_in', 'Y4_in', 'Y5_in', 'Y6_in', 'Y7_in']].mean()




    for sns in ['fb', 'tw', 'in']:

        print('=========={}===========\n'.format(sns))
        for i in range(1, 8):
            col_name = 'Y{}_{}'.format(i, sns)
            average = df_ave[col_name]
            y = data[col_name]
            model = sm.OLS(y, big5)
            results = model.fit()
            print(col_name.replace("Y" + str(i), sns_characteristics[i - 1]))
            print('mean: {0:.3f}'.format(average))

            print(pd.DataFrame([results.params, results.pvalues], index=['Para', 'pvalue']))
            print('R2: {0:.3f}'.format(results.rsquared))
            print('\n--------------------\n')



if __name__ == '__main__':
    main()
