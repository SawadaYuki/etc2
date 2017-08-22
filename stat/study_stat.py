# -*- coding: utf-8 -*-

import xlrd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import integrate
from scipy import stats
import numpy.random
from numpy.random import *
import itertools # conbination

print_flag = 0

def read_Data(str):
	book = xlrd.open_workbook(str)
	sheet1 = book.sheet_by_index(0)

	array = []
	line = []
	

	for row_index in range(sheet1.nrows):
	    for col_index in range(sheet1.ncols):
	        val = sheet1.cell_value(rowx=row_index, colx=col_index)
	        # print val
	        # print type(val)
	        utf_val = val.encode('utf-8')
	        #print type(utf_val)
	        tmp_val = float(utf_val)
	        #print type(tmp_val)
	        if isinstance(tmp_val,float):
	        	line.append(tmp_val)
	        	#print line
	    
	    if len(line) != 0:
			array.append(line)
			line = []

	
	# debug print
	# print len(array)
	# array.reverse()


	# while len(array) != 0:
	# 	print array.pop()

	return array

def debug_print(data,data_size,data_line_size):
	print "data size = " + str(data_size)
	print "1 data size = " + str(data_line_size)
	print data[0]
	i = 0
	
	while i < data_line_size :
		print data[0][i]
		print type(data[0][i])
		i = i + 1

#引数:観測データの塊,抽出したい1データ項目の列番某
def extract_one_column(data,column_index):
	#print column_index
	column = []
	#print len(data)
	i = 0

	while i < len(data) :
		column.append(data[i][column_index])
		#print "i = "+str(i) + ", val = " + str(column[i])
		i = i + 1

	
	return column

#算術平均の取得
def get_arithmetic_mean(data):
	total = 0
	i = 0
	while i < len(data) :
		total = total + data[i]
		i = i+1

	#print "(算術平均) total = " + str(total) + ", N = " + str(len(data)) + ", arithmetic_mean = " + str((total / len(data) ) )
	return (total / len(data) ) 

#調和平均 
def get_harmonic_mean(data):
	
	Explanatory_text = ("調和平均 ... 速度(道のり/時間)のように、比率で考えられる量の平均を求める場合に活用される"
		"1/x =  (1/n) * (1/x1 + 1/x2 + ...)"
		)
	if print_flag == 1 :
		print Explanatory_text 
	
	total = 0
	i = 0
	while i < len(data) :
		total = total + (1/data[i])
		i = i+1

	print "(調和平均) total = " + str(total) + ", N = " + str(len(data))
	print  (1/len(data) * total) 
	return (1/len(data) * total)

#中央値
def get_median(data):
	N = len(data)
	at_25 = int(N/4) # 25%地点
	at_middle = int(N/2) # 50%地点
	at_75 = int(N * 3/4) # 75%地点

	data.sort()

	print "[25%地点]"+str(data[at_25])
	print "[50%地点]"+str(data[at_middle])
	print "[75%地点]"+str(data[at_75])

	return data[at_middle]

#四分位偏差Q
def get_quartile_deviation(data):
	data.sort()
	N = len(data)
	at_25 = int(N/4) # 25%地点
	at_75 = int(N * 3/4) # 75%地点

	#print "[25%地点]"+str(data[at_25])
	#print "[75%地点]"+str(data[at_75])
	
	Q = (data[at_75] - data[at_25]) / 2
	Explanatory_text = ("Qが大きいほど、散らばった分布である")
	if print_flag == 1:
		print Explanatory_text

	return Q

#引数:一次元のデータ配列
def print_hist(one_dgree_data):
	Explanatory_text = ("適した階級数 = 1 + log2n (スタージェスの公式)"
		"2つ以上峰がある場合は、性質の異なるデータが混じりあっているので「層別」が必要"
		)
	# データ数 = n
	print len(one_dgree_data)
	# bins = 階級数
	n, bins, patches  = plt.hist(one_dgree_data, bins=10)
	plt.show()
	if print_flag == 1:
		print Explanatory_text


#分散
def get_variance(data) :
	arithmetic_mean = get_arithmetic_mean(data)
	N = len(data)
	i = 0
	total = 0
	while i < N :
		deviation = arithmetic_mean - data[i] # 偏差
		total = total + pow(deviation,2)
		i = i + 1
	return total / N

# 標準偏差
def get_standard_deviation(variance):
	return math.sqrt(variance)

def print_basic_info(data):
	arithmetic_mean = get_arithmetic_mean(data)
	#平均値より大きい/小さいデータがたくさんある場合、「平均値」は代表値としてふさわしくない
    #=>中央値を活用する
	median = get_median(data)
	
	#データの散らばりをざっくり見たい
    #=>四分位点を活用
	quartile_deviation = get_quartile_deviation(data)
	
	#↑は25%地点と75%地点の値の2つしか使わず、分布の散らばり具合を見ている
    # => 全データを活用して散らばり具合を見る
	variance = get_variance(data)
	standard_deviation = get_standard_deviation(variance)

	print "+++++++++++++++++"
	print "データ数:" + str(len(data))
	print "算術平均:" + str(arithmetic_mean)
	print "中央値:" + str(median)
	print "四分位偏差:" + str(quartile_deviation)
	print "分散:" + str(variance)
	print "標準偏差:" + str(standard_deviation)
	print "+++++++++++++++++"
	
	#print 度数分布
	#print_hist(data)

#変動係数
def compare_coefficient_of_variation(data1,data2):
	arithmetic_mean1 = get_arithmetic_mean(data1)
	arithmetic_mean2 = get_arithmetic_mean(data2)
	standard_deviation1 = get_standard_deviation(get_variance(data1))
	standard_deviation2 = get_standard_deviation(get_variance(data2))

	C_V1 = standard_deviation1 / arithmetic_mean1
	C_V2 = standard_deviation2 / arithmetic_mean2

	print "C_V1:" + str(C_V1) 
	print "C_V2:" + str(C_V2)

def Print_scatter_plot(x,y):

	plt.scatter(x, y)
	plt.show()

#相関係数(最もよく使われるピアソンの積率相関係数)
def get_correlation_coefficient(dataSet1,dataSet2):
	standard_deviation_x = get_standard_deviation(get_variance(dataSet1))
	standard_deviation_y = get_standard_deviation(get_variance(dataSet2))


	arithmetic_mean_x = get_arithmetic_mean(dataSet1)
	arithmetic_mean_y = get_arithmetic_mean(dataSet2)
	i = 0
	if len(dataSet1) != len(dataSet2):
		print "2変数のデータ数が異なります"
		return -1

	N = len(dataSet1)
	covariance=0
	while i < N:
		covariance = covariance + ((dataSet1[i] - arithmetic_mean_x) * (dataSet2[i] - arithmetic_mean_y))
		i = i + 1
	covariance = covariance / N
	
	correlation_coefficient = covariance / (standard_deviation_x * standard_deviation_y)

	print "相関係数 = " + str(correlation_coefficient)

	if correlation_coefficient == 1 :
		print "強い正の相関"
	elif 0 < correlation_coefficient <1 :
		print "正の相関"
	elif correlation_coefficient == 0 :
		print "相関なし"
	elif -1<correlation_coefficient<0 :
		print "負の相関"
	elif correlation_coefficient == -1 :
		print "強い負の相関"
	else :
		print "無効な相関係数です"

	
	return correlation_coefficient

# 偏相関係数
def get_partial_correlation_coefficient(dataSet1,dataSet2,dataSet3):
	r_xy = get_correlation_coefficient(dataSet1,dataSet2)
	r_xz = get_correlation_coefficient(dataSet1,dataSet3)
	r_yz = get_correlation_coefficient(dataSet2,dataSet3)

	r_xy_z = (r_xy - r_xz*r_yz) / (math.sqrt(1-pow(r_xz,2)) * math.sqrt(1-pow(r_yz,2)))
	print "偏相関係数 = " + str(r_xy_z)
	return r_xy_z

#　遅れhの自己相関係数
def get_auto_correlation_coefficient(data,h):
	data2 = []
	i = 0
	N = len(data)
	r_h = 0.0
	arithmetic_mean = get_arithmetic_mean(data)
	variance = get_variance(data)

	tmp_val = 0.0
	
	while i < N - h :
		#遅れhのデータセットを用意
		data2.append(data[h])
		tmp_val = tmp_val + (data[i] - arithmetic_mean)*(data2[i] - arithmetic_mean)
		i = i + 1

	r_h = (tmp_val / (N-h)) / variance

	if r_h < 0:
		print "地点「" + str(h) + "」後には値が反転する様子が見られる"
	elif r_h > 0:
		print "地点「" + str(h) + "」後にはその値の傾向が継続する様子が見られる"
	else :
		print "自己相関係数 = " + str(r_h)

	return r_h

#回帰方程式を決定づける偏回帰変数bとy切片aを求める
def get_regression_line(x,y,b,a):
	print "python lib ... scipy.stats.linregress "

	#最小二乗和でb,aを求める

	#初期化
	L = 0.0
	i = 0
	b = 0.0
	a = 0.0

	N = len(x)
	arithmetic_mean_x = get_arithmetic_mean(x)
	arithmetic_mean_y = get_arithmetic_mean(y)

	while i < N :
		# y[i]が実際の値 VS 「b*x[i] + a」が予測値
		L = L + pow( (y[i] - (b*x[i] + a)) , 2 )
		i = i + 1

	# 得られた「L」を最小化するb,aを求める
	# Lはa,bの二次方程式だから、b,aでそれぞれ偏微分して0とおいて、解く(省略)	
	# b = 共分散 / 偏差の二乗和
	i = 0
	val = 0
	while i < N :
		b = b + ((x[i] - arithmetic_mean_x)*(y[i] - arithmetic_mean_y))
		val = val + pow((x[i] - arithmetic_mean_x),2)
		i = i+1

	b = b / val

	a = arithmetic_mean_y - (b * arithmetic_mean_x)
	# 傾き, 切片, 相関係数, 仮説検定のための両側p値, 見積もりの​​標準誤差
	slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
	print "slope:" + str(slope)
	print "intercept:" + str(intercept)
	print "(回帰方程式)y = " + str(b) + "x + " + str(a)

	#求めた回帰方程式は実際に観測された値とどれほど近いのか?
	#相関係数が当てはまりの良さの尺度となる
	#d:誤差 d = y[i] - (b*x[i]+a)
	
	i = 0
	total_d = 0.0
	while i < N :
		d = y[i] - (b*x[i]+a)
		d_2 = pow(d,2)
		total_d = total_d + d_2
		i = i + 1

	sum_deviation = get_variance(y) * N # = 分散 * N = 偏差の二乗和
	r_xy = get_correlation_coefficient(x,y)

	# ここの「=」は代入ではなく、等しいの意
	# 下記の式から、r^2が1に近くほど誤差が0,つまり実測値に近い
	# => r^2は説明変数xがyを決定する強弱の度合いを表している(r^2:決定係数)
	# おまけ: b = r * (standard_deviation_y / standard_deviation_x)
	total_d = (1-pow(r_xy,2)) * sum_deviation 

	print "当てはまりの良さ(相関係数) = " + str(r_xy) + ", 誤差の二乗和 = " + str(total_d)





#1変数(データ)の分析
def analysis_one_variable(data):
	#自己相関係数
	# = 同じ変数でも系列(時系列)の異時点間の相関を見たい
	h = 3
	get_auto_correlation_coefficient(data,h)

#2変数(データ)の分析
def analysis_two_variable(data1,data2):
	print "analysis two variable"
	#2つの標準偏差を比べ,どちらの散らばり具合が上か下か
	#=>両者の平均が大きく異なる場合、単純に両者を比較できない
	#=>変動係数を活用する
	compare_coefficient_of_variation(data1,data2)

	#散布図を表示
	Print_scatter_plot(data1,data2)

	#相関係数(量的データ)
	correlation_coefficient = get_correlation_coefficient(data1,data2)
	if correlation_coefficient != -1 :
		print "相関係数 = " + str(correlation_coefficient)
    	#相関関係と因果関係は異なる = 一方がもう一方を決めるとは言えない
    	# ex) 身長と体重 ... 相関はあるが、身長が体重を決めるわけではない
    	# また、「見かけ上の相関(変数Zを挟んで、変数xと変数yの間に相関がある)」の場合がある
    	# => 変数Zの影響を除いた相関係数(偏相関係数)を求めたい場合
    	# ==> analysis_three_variable(data1,data2,data3)

	#順位相関係数(質的データ) ... ToDo
	#xとyの関係を「回帰」という方法で扱う 
	#(回帰方程式) y = bx + a ... y:被説明変数,b:偏回帰変数,x:説明変数
	b = 0.0
	a = 0.0
	get_regression_line(data1,data2,b,a)



#3変数(データ)の分析
def analysis_three_variable(data1,data2,data3):
	print "analysis three variable"
	#偏相関係数
	get_partial_correlation_coefficient(data1,data2,data3)

def sigma(a,b,func,total = 0):
	if a>b : return total
	return sigma(a+1,b,func,total + func(a)) 

# 確率について
def print_Explanatory_text_about_probability():

	#確率論
	probability_theory_text = ("全体としては、ランダムネス(次に何が起こるか予想できない)には法則性があるため、確率論が成り立つ"
		"事象の起こりやすさを定量的に示したい => ラプラスの定義"
		"ラプラスの定義:標本空間内の各事象は、同程度に確からしいと仮定。確率 = 事象Aの起こるような根元事象の数/試行の根元事象の総数"
		"<=反対の十分な理由はないため、仮定しても妥当であると信じられる")
	print probability_theory_text

	#確率の頻度説
	frequency_theory_text_1 = ("「同程度に確からしい」と仮定を与えられない場合 => 確率の頻度説。"
		"事象Aを生み得る実験をn回繰り返して、事象AがnA回出るとする。"
		"(確率の頻度説を適用)n->「無限」のとき、nA/n -> αとなるならば、「P(A) = α」と定義。これは観測データに基づいた確率となる。"
		"(しかし、極限への収束は無限に試行して、初めて確認される)"
		"数学的なモデルとして体型的に表すために、「確率」を公理として定義する。"
		)
	print frequency_theory_text_1
	#公理 ... 無条件で認められた事実
	#定理 ... 公理から導き出され、定義された言葉のみで構成された正しいことが証明できる文章 

	frequency_theory_text_2 = ("(1).全ての事象Aに対して、0<= P(A) <= 1 "
		"(2).P(Ω) = 1 "
		"互いに排反な事象A1,A2...に対してP(A1 U A2 U ...) = P(A1) + P(A2) + ... "
		"(1) ~ (3)を満たすならどのような数も確率と認める")
	print frequency_theory_text_2

def print_Bayes_theorem() :
	Bayes_theorem_text = ("起こった事象Aの原因が事象Hiである確率は、P(Hi|A)であるが、"
	"知ることが出来るのは原因に対する結果の確率 P(A|Hi)である場合がほとんど。"
	"前提条件 H1,H2, ... は互いに排反かつ H1 U H2 ... = Ω 。"
	"ベイズの定理　P(Hi|A) = P(Hi)*P(A|Hi) / Σ (P(Hj)*P(A|Hj)) 。"
	"outputは「結果の原因が事象Hiである確率」")
	print Bayes_theorem_text

	#おまけ:条件付き確率
	#ほかの事象Bが起こったと分かっている場合に、事象Aが起こる確率 = 事象Bを条件とする事象Aの条件付き確率 = P(A|B)
	# P(A|B) = P(AかつB) / P(B)
	# ここで、そもそも事象Bの起こる確率が他の事象に影響するかどうか
	# => 独立ならば P(A) = P(A|B) ... つまり事象Bが起こる/起こらないに関わらず事象Aの起こる確率はP(A)

# 確率変数の基礎
def probability_variable(y,a,b):
	
	#確率変数 ... それがとる値にそれぞれ確率が付与されている変数
	# Ex) X=サイコロの目
	# X   | 1   | 2 | 3 .... |
	# -------------------------
	# P(X)| 1/6 |1/6|1/6 ....|

	# f(x)
	f_x = y


	# もし連続型の確率変数の確率分布の場合
	# loc:期待値,scale:標準偏差,size:取得する個数
	print "python lib ... scipy.stat.norm.rvs(loc=0, scale=1, size=1, random_state=None)"
	# P(a<= x <= b) = ∫ f(x)dx
	# 定理)
	#	全てのxに対し,f(x) >= 0 かつ ∫ f(x)dx = 1 [-∞,∞]
	#	一点の確率は0となる(これは離散型と異なる点である)
	P_a_to_b = integrate.quad(f_x,a,b)

	# もし離散型の確率変数の確率分布の場合
	# おまけ)一般に、可算集合{x1,x2 ... }の中の値をとる確率変数Xは離散型と呼ばれる
	# P(X = xk) = f(xk) k = 1,2,... (※ f(xk)は確率分布
	# 定理)
	#  f(xk) >= 0 , Σf(xk) = 1 [1,∞]

# 累積分布関数
def cumulative_density_function():
	print "ある値以下の確率を求めたい場合に活用"
	print "python lib ... scipy.stat.norm.cdf(x, loc=0, scale=1)"

	# P(X <=x ) = F(x)

	# 連続型の場合: ∫ f(u)du = F(x) [-∞, x] <=密度関数fの定積分
	# 離散型の場合: Σ f(u) = F(x) [-∞ < u <= x] 
	# 累積分布関数の性質
	# 	x1 < x2ならば　F(x1) <= F(x2)
	#	x => ∞のとき、F(x) -> 1 , x => - ∞のとき F(x) -> 0
	#	各店xでε(限りなく0に近い)とき、F(x + ε) -> F(x) ... つまり整数店で不連続点

# 期待値
def expected_value(f,n,x1= 0,x2=0):
	print "色々な値をとる確率変数において、客観的かつそれらの値を代表する値を知りたい場合は期待値"
	E_X = 0.0

	# 確率変数Xが離散型: E(x) = Σ xf(x)	
	x = 1
	N = n 
	
	while x <= N:
		#print "f(x) = " + str(f(x))
		E_X = E_X + x*f(x)
		x = x + 1

	print "離散型の確率変数の期待値E(x) = " + str(E_X)

	# 確率変数Xが連続型: E(x) = ∫ xf(x) [-∞,∞]
	a = -np.inf
	b = np.inf
	if x1 != -np.inf:
		a = x1
	if x2 != np.inf:
		b = x2
	print "f(x)の値域[" + str(a) + "," + str(b) + "]"
	E_X,abserr = integrate.quad(lambda x: x*f(x),a,b)
	print "連続型の確率変数の期待値E(x) = " + str(E_X) + ",abserr = " + str(abserr)

	#　期待値の演算の性質
	theory = (" (a) E(c) = c "
		"(b) E(x+c) = E(x) + c "
		"(c) E(cx) = cE(x) "
		"(d) E(x+y) = E(x) + E(y)"
		)
	print theory

	# しかし、期待値E(x)が同じでも確率分布の様子が異なるものもある
	# => 重要な指標に「バラツキ」がある

	# 分散V(x) = E{(X-μ)^2} ... μ = E(x)
	#    V(x) = E(x^2) -2μE(x) + μ^2 = E(x^2) - μ^2 = E(x^2) - (E(x))^2
	# 標準偏差 D(x) = √V(x)

	# 観測データの標準化と同様、確率変数Xを標準化して、他の確率変数と比較できる
	# Z = {X - E(x)} / √V(x) ... 確率変数Zは期待値=0,分散=1,標準偏差 = 1

# 確率分布の代表値
def probability_distribution_representative_values():
	# 期待値,分散,歪度,尖度
	# σ^2 = 分散V(x)

	# 歪度: α3 = E(x-μ)^3 / σ^3 ... α3 >0 ならば右の裾が長い分布
	print "python lib ... scipy.stats.skew(data, axis=0, bias=True, nan_policy='propagate') = ndarray(歪度)"

	# 歪度:α4 = E(x-μ)^4 / σ^4 ... 普通は正規分布のα4 = 3と比較して、「α4-3」を「Xの確率分布の尖度」
	# 「α4-3」 > 0 ならば正規分布より尖っている
	print "python lib ... scipy.stats.kurtosis(data, axis=0, fisher=True, bias=True, nan_policy='propagate')"
	# => fisher ... trueならフィッシャーの定義が使用され,flaseならピアソンの定義が使用

	#上記から、確率分布の形はE(X-u)^r　なる量で決まってくることが分かる
	# μr = E(X^r) ... xの原点周りのr次モーメント
	# (μr)' = E(X-μ)^r ... xの期待値周りのr次モーメント
	# => モーメントの集合内{μ1 = E(x) = 期待値, μ2' = E(X-μ)^2 = 分散 ... }
	# おまけ)
		# 統計学におけるモーメントは物理学におけるモーメントの類推である．
		# 物理学におけるモーメントが長さと力の積であるのに対し，統計学のモーメントは標本と確率の積で与えられる．
		# 確率変数Xの原点まわりの r次のモーメント μr は以下で定義される．
		# 原点まわりの1次のモーメントは，まさに確率変数X の期待値そのものである．
		# 2次，3次および4次のモーメントはそれぞれ，分散，歪度および尖度の計算に用いられる．
	
	# モーメント母関数 = 確率分布そのもの　 ... 1つの確率分布が決定する
	# モーメント母関数 Mx(t) = E(e^tx) ... 繰り返し微分して0とおいた導関数Mx'(0),Mx''(0)...を解き、各字数のモーメントを明らかにしていく
	# => 離散型 Mx(t) = Σe^tx * f(x)
	# => 連続型 Mx(t) = ∫ e^tx * f(x)dx

#チェビシェフの不等式 ... どんな確率変数についても成立する絶対的な式
def Chebyshev_inequality():
	print "平均や分散といった代表値しか分からず、確率分布もわかっていない"
	print "その中で確率/確率分布の見当をつけたい!"

	# input: μ = E(x), σ^2 = V(x)　<=この2つの値だけで事足りる
	# チェビシェフの不等式=>  P(|X-μ| >= kσ) <= 1/k^2
	# ex) P(0<= X <= 2)の値を求めることができる

#超幾何分布 ... 「全体数」を十分大きくすると、二項分布に近づく
def hypergeometric():

	# 2種類の要素から構成させる母集団があり、AがM個,BがN個あるとする
	# そこからnsamp個取り出す(非復元)
	# n個のうち、Aがx個含まれる確率 f(x) = Combination(M,x) * Combination(全体-M,n-x) / Combination(全体,N)
	# このf(x)が従う分布を超幾何分布
	# 比例式 全体:M = n:xが成立する場合、「全体」を推量できる
	# E(X) = n * (M/全体), V(X) = n * (M/全体) * (1 - (M/全体)) 

	# 超幾何分布から生成される乱数を生成する。
	# そして例えばM個の良品とN個の不良品があって、そこからnsamp個を不良率調査で抜き出した時に取り出せた良品の個数を返す。
	M, N, nsamp = 90, 10, 10
	x = numpy.random.hypergeometric(M, N, nsamp, 100)
	print x
	print np.average(x)

# ベルヌーイ試行
def Bernoulli_trials():
	s = ("2種類の可能な結果(例えば、成功と失敗)を生じる実験を、同じ条件かつ独立に試行したもの")
	print s 

# 二項分布
def Binomial_distribution():
	# ベルヌーイ試行をn回繰り返した場合の確率分布を二項分布 Bi(n,p)

	# p:1回の試行において、Sとなる確率
	# 1-p:Fとなる確率
	# n:回数
	# x:S/Fが得られた回数
	# 二項分布 Bi(n,p) ... nやpのように、分布を特徴づける値を「母数」という
	# P(X = k) = f(k)
	# f(x) = Combination(n,x)*p^x * (1-p)^n-x
	# E(x) = np, V(x) = np(1-p)

	# ex) 確率pでオモテが出るコインをn回投げて、オモテが出る個数の分布から生成される乱数を生成する。
	x = binomial(n=100, p=0.5,size = 300)
	E_X = np.mean(x)
	V_X = np.var(x)
	print "E(x):" + str(E_X) + ", V(X):" + str(V_X)
	plt.hist(x, 10)
	plt.show()

#ポアソン分布 ... 二項分布において、nが大きくPが小さい(つまり稀な現象)の場合に「np=一定」と考えたもの
def Poisson_distribution():
	# ポアソンの小数の法則
	# => E(x) = np -> λとなるように, n -> ∞, p->0となるような極限では、各xについて
	#	「f(x) = Combination(n,x)p^x * (1-p)^n-x  -> (e^-λ * λ^x) / x! 」が成り立つ = ある期間に平均λ回起こる事象がx回起こる確率
	#	E(x) = λ, V(x) = λ

	# λ=10 のポアソン分布 ... 稀にしか起きない現象を長時間観測したときに起きる回数の分布
	x = poisson(lam = 10, size = 30)
	print x
	E_X = 10
	V_X = 10
	print "E(x):" + str(E_X) + ", V(X):" + str(V_X)
	plt.hist(x, 10)
	plt.show()

# 幾何分布 ... ベルヌーイ試行の回数を設定しないで、最初の成功(=事象が起こる)まで試行を続け、ある事象が起こるまでの時間(試行回数)の分布
def geometric_distribution():
	# x:試行回数
	# 失敗する確率 q = 1-p 
	# => f(x) = p*q^x-1
	#	E(x) = 1/p ... 1回ある事象が起きるためには平均1/p岳の時間を要することが分かる 
	#	V(x) = q/p^2

	p = 0.5 # 1回の試行における成功確率
	q = 1-p 
	sample = numpy.random.geometric(p = 0.5 , size = 100)
	E_X = 1/p
	V_X = q/pow(p,2)
	print "E(x):" + str(E_X) + ", V(X):" + str(V_X)
	plt.hist(sample, 10)
	plt.show()

	print "幾何分布はある事象が1回起きるまでに要する待ち時間分布だが、ある事象がK回起きるまでに要する待ち時間分布を「パスカル分布」という"


# 正規分布 ... 代表的な連続型の確率分布
def nomal_distribution():
	# 自然界や人間社会の中の数多くの現象に対して当てはまるのが「正規分布」
	# 試行回数nが大きくなるときに、「正規分布が出現することは中心極限定理により、示されている」

	#　正規分布N(μ,σ^2)
	#  => f(x) = 1/√2πσ * exp{-(x-μ)^2/2σ^2} -∞ < x < ∞
	#		E(x) = μ
	#		V(x) = σ^2
	mean = 10
	variance = 2
	sample = np.random.normal(mean,pow(variance,2),size = 10000)
	#print sample
	E_X = mean
	V_X = pow(variance,2)
	print "E(x):" + str(E_X) + ", V(X):" + str(V_X)
	plt.hist(sample, 10)
	plt.show()

	# 1) Xが	N(μ,σ^2)に従っているとき、その線形変換 Y = aX + bは,N(aμ+b, a^2 * σ^2)に従う
	# 2) 標準化変数Z = (X-μ)/σは、標準正規分布 N(0,1)に従う
	#	P(-l <= Z <= k) = P(Z<=k) - P(z<-k) 
	#	P(-1 <= Z <= 1) = 0.68
	#	P(-2 <= Z <= 2) = 0.95
	#	P(-3 <= Z <= 3) = 0.9973  ...3シグマ以内にデータの99%が存在することが分かる
	#   => 元のxに変換すると, μ - 3σ <= x <= μ + 3σ 内にデータの99%が存在することが分かる
	


#指数分布 Ex(λ)
def exponential_distribution():

	# f(x) = λ * exp(-λx) , x >= 0
	# f(x) = 0  , x<0
	# 上記の確率密度関数で定義される連続型の分布である

	# ところで、幾何分布において、 q=e^-λとすると、指数分布の主要な部分exp(-λx)が得られるから、
	# この確率分布は連続的な「待ち時間分布」の性質を持つことが分かる
	# E(x) = 1/λ, V(x) = 1/λ^2

	# pが小さい幾何分布においてはE(X) ≒ D(x) 標準偏差
	# 指数分布では正確に E(x) = D(x)となる。そのためE(x)±D(x) = 0 ~ E(x) + D(x)となり、0を範囲に含む
	# =>  そのため、指数分布によって、生起までの年数が分布する希少事象は近い将来起きても不思議ではない = 確率pが小さいと遠い将来にしか起こらないということではない
	
	# scale = 1/λ
	scale = 1 
	sample = np.random.exponential(scale,size = 10000)
	print sample
	E_X = scale
	V_X = pow(scale,2)
	print "E(x):" + str(E_X) + ", V(X):" + str(V_X)
	plt.hist(sample, 10)
	plt.show()	 

#ガンマ分布 ... 指数分布を一般化したもの
def gamma_distribution():
	# ガンマ分布 Ga(α,λ)
	# f(x) = (λ^α / Γ(α)) * x^(α-1) * exp(-λx) , x>= 0
	# f(x) = 0 , x<0
	# α = 1ならば指数分布となる
	# おまけ) 積分して1となるガンマ関数 ... Γ(α) = ∫ x^(α-1)*exp(-x)dx [0,∞] 
	# E(x) = α/λ , V(x) = α / λ^2

	sample = np.random.gamma(shape = 1,scale = 2,size = 100)
	plt.hist(sample, 10)
	plt.show()
       

# ベータ分布...(0,1)上の確率分布 Be(α,β)
def Beta_distribution():
	# 現象例は少ないが,ベイズ統計学での役割は大きい
	# α,βの値によって色々な形をとる

	# f(x) = x^(α-1) * (1-x)^(β-1) / B(α,β)　, 0<x<1
	# f(x) = 0 , x<=0 かつ x>=1
	# E(x) = α / (α + β), V(x) = αβ / {(α+β)^2 * (α+β+1)}
	# おまけ) B(α,β) = ベータ関数 = ∫ x^(α-1) * (1-x)^(β-1) dx [0,1] α > 0 β > 0
	sample = np.random.beta(a = 1,b = 1,size = 100)
	plt.hist(sample, 10)
	plt.show()

# コーシー分布 ... 期待値も分散も存在しない正規分布に似た分布
def cauchy_distribution():
	# モーメント母関数も存在しない
	# f(x) = α / π {α^2 + (x-λ)^2} α > 0
	sample = numpy.random.standard_cauchy(size = 10000)
	plt.hist(sample, 10)
	plt.show()

# 対数正規分布
def log_nomal_distribution():
	# 世帯所得のように,低い方は一定限度があるが、高い方には明確な限度がない場合、log(X)を考えるのが自然である
	# log(X)が正規分布に従うならば,元のXは対数正規分布に従うという

	# f(x) = 1/(√2π * σx) * exp{-(logx - μ)^2 / 2σ^2} x> 0
	# f(x) = 0 x<=0

	# E(x) = exp(μ + σ^2 / 2)
	# V(x) = exp(2μ + 2σ^2) - exp(2μ + σ^2)
	sample = numpy.random.standard_cauchy(mean = 5, sigma = 10, size = 10000)
	plt.hist(sample, 10)
	plt.show()

#パレート分布
def pareto_distribution():
	# 対数正規分布が全集団の所得分布に対して、パレート分布は高額所得者の所得分布である
	# x0以上の所得の確率は、対数分布よりもこの分布がよく当てはまる
	# => ある定数x0以上で存在し,そこから急に減少する密度関数を持つ
	# f(x) = (a/x0) * (x0/x)^(a+1) x >= x0
	# f(x) = 0 x < x0
	# E(x) = ax0 / (a-1)   ...  (a>1)
	# V(x) = {ax0^2 / (a-2)} * {ax0 / (a-1)}^2
	sample = np.random.pareto(a=2.718281828, size=10000)
	plt.hist(sample, 10)
	plt.show()



















	















