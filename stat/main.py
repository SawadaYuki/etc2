# -*- coding: utf-8 -*-
import study_stat
from scipy import integrate

def f(x):
	return 2*x

def f1(x = 0):
	return 0.16666


if __name__ == '__main__':
	
	print_flag = 1
	data = study_stat.read_Data('fitbit_export_20170815.xls')
	data_size = len(data)
	data_line_size = len(data[0])
    #debug_print(data,data_size,data_line_size)

	column1 = study_stat.extract_one_column(data,0)
	study_stat.print_basic_info(column1)

	column2 = study_stat.extract_one_column(data,1)
	study_stat.print_basic_info(column2)

 	column3 = study_stat.extract_one_column(data,2)
	study_stat.print_basic_info(column3)

	#analysis_one_variable(column1)
	#study_stat.analysis_two_variable(column1,column2)
	#analysis_three_variable(column1,column2,column3)

	#print_Explanatory_text_about_probability()

	#また起こっていないかほとんど起こっていない事象や実験ごとに統計的規則が変わるような事象を分析したい
	# => 「主観説の立場」からベイズの定理を用いて展開(ベイズ統計学) <=> 「客観説の立場」 ... 誰が計算しても同一の値(例えば、ラプラスの定義や頻度説)
	
	#print_Bayes_theorem()
	

	#study_stat.expected_value(f1,6,0,6)
	#study_stat.hypergeometric()
	#study_stat.Binomial_distribution()
	#study_stat.Poisson_distribution()
	#study_stat.geometric_distribution()
	#study_stat.nomal_distribution()
	#study_stat.exponential_distribution()
	#study_stat.gamma_distribution()
	#study_stat.Beta_distribution()
	study_stat.cauchy_distribution()

