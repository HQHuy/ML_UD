# nhập thư viện
import numpy as np
import pandas as pd

# tạo hàm lấy dữ liệu
def loadCsv(filename) -> pd.DataFrame:
    data = pd.read_csv(filename)
    return data

# tạo hàm chia dữ liệu thành train và test, tạo X_train/y_train và X_test/y_test.
def splitTrainTest(data, ratio_test):
    np.random.seed(28) # không thay đổi mỗi lần chạy
    index_permu = np.random.permutation(len(data)) # xáo trộn index 
    data_permu = data.iloc[index_permu] # lấy lại dữ liệu tương ứng với index xáo trộn
    len_test = int(len(data_permu)*ratio_test) # kích cỡ tập test
    test_set = data_permu.iloc[: len_test, :] # tập test lấy phần đầu
    train_set = data_permu.iloc[len_test: , :]  # tập train lấy phần còn lại
    # Chia tập dữ liệu thành (x_train, y_train: Lấy tất cả cột trừ dòng cuối), (x_test, y_test: Chỉ lấy cột cuối)
    '''Code here'''
    X_train = train_set['Text']
    y_train = train_set['Label']
    X_test = test_set['Text']
    y_test = test_set['Label']
    '''END'''
    return X_train, y_train, X_test, y_test

# hàm lấy tần số của từ: 1. lấy túi từ (bag words), 2. lấy tần số từ
def get_words_frequency(data_X):  # tạo hàm lấy tần số từ, data_X --> DataFrame
    bag_words = np.concatenate([i[0].split(' ') for i in data_X.values], axis = None)
    # B1: lấy các giá trị của DataFrame --> Array
    # B2: lặp qua các phần tử trong mảng chính là i --> string (text)
    # B3: lấy các từ trong đoạn text ra --> [['VKE', 'đánh', 'CKTG', 'vậy', 'là', 'hay', 'rồi'], ['Đã', 'quá', 'VKE', 'ơi']]
    # B4: hàm concatenate với tham số axis = None có chức năng làm phẳng và đưa về mảng một chiều
    # --> array['VKE', 'đánh', 'vậy', 'là', 'hay', 'rồi', 'Đã', 'quá', 'VKE', 'ơi']
    
    bag_words = np.unique(bag_words) # loại bỏ các giá trị trùng và lấy giá trị duy nhất trong mảng bag_words
    matrix_freq = np.zeros((len(data_X), len(bag_words)), dtype = int) # tạo ma trận 0 có kích cỡ [số dòng data_X(dòng) x số từ trong túi từ(cột)]

    word_freq = pd.DataFrame(matrix_freq, columns = bag_words) # tạo frame với matrix_freq, cột là các từ trong túi từ 
    for id, text in enumerate(X_train.values.reshape(-1)):
    # hàm enumerate sẽ gán index cho mỗi phần từ, có tham số start để có thể điều chỉnh số bắt đầu
    # index lưu vào id, phần tử lưu vào text
        for j in bag_words: # đối với mỗi id (dòng), ta lặp qua các từ trong túi (cột)
            word_freq.at[id, j] = text.split(' ').count(j) # đếm từ đó có trong biến text và gán tại vị trí [id, j]
    return word_freq, bag_words # trả lại biến tần số từ, --> DataFrame(cột là các từ trong túi từ)

def transform(data_test, bags): # bags là bag_words được return từ hàm get_words_frequency, data_test dạng frame
    matrix_0 = np.zeros((len(data_test), len(bags)), dtype = int)
    frame_0 = pd.DataFrame(matrix_0, columns = bags)
    for id, text in enumerate(data_test.values.reshape(-1)):
        print(text)
        for j in bags:
            frame_0.at[id, j] = text.split(' ').count(j)
    return frame_0

# trước khi tính khoảng cách thì đổi tập test sang số.

# tạo hàm tính khoảng cách: Euclid, Manhattan, Cosine
# khoảng cách Cosine: đưa data_X về Array để tính không còn DataFrame

# Bên trên X_train và X_test ở dạng DataFrame, ta phải cho đi qua hàm lấy tần số từ, kết quả trả về vẫn là DataFrame có giá trị là các số
# Chuyển DataFrame này về dạng mảng trước khi đưa vào hàm tính khoảng cách dùng values --> mảng nhiều chiều 
def cosine_distance(train_X_number_arr, test_X_number_arr): 
    dict_kq = dict() # tạo dictionary trống
    for id, arr_test in enumerate(test_X_number_arr, start = 1):
    # tương tự lấy index cho mỗi phần tử trong mảng test_X_number_arr, index đánh bắt đầu bằng 1
        
        q_i = np.sqrt(sum(arr_test**2)) # căn của tổng ([q_i]^2), dùng để tính mẫu
        for j in train_X_number_arr:
            _tu = sum(j*arr_test) # tính tử: tổng q[i]*dj[i]

            # tính mẫu: (căn của tổng (q[i]^2)*(căn của tổng (dj[i])^2)
            d_j = np.sqrt(sum(j**2))
            _mau = d_j*q_i

            # kết quả: lấy tử chia mẫu --> khoảng cách của mỗi dòng trong test_X với các dòng trong train_X
            kq = _tu/_mau 
            
            # nếu index có trong dict_kq rồi thì ta thêm giá trị kq vào, nếu chưa thì ta tạo khoá id với giá trị kq.
            if id in dict_kq:
                dict_kq[id].append(kq)
            else:
                dict_kq[id] = [kq]
    
    return dict_kq # --> Dictionary với key: dòng trong tập test, value: các giá trị đã được tính khoảng cách với các dòng trong tập train
    # ví dụ: {1: [2, 3, 4, 5, 6]}, 1 là dòng thứ nhất trong tập test, [2, 3, 4, 5, 6] là khoảng cách của dòng 1 trong tập test đến các dòng trong tập train

# lớp KNN
class KNNText:
    # hàm tạo
    def __init__(self, k): # k là số điểm dữ liệu gần nhất
        '''Code here'''
        self.k = k 
        '''END'''
    # hàm fit
    def fit(self, X_train, y_train):
        '''Code here'''
        self.X_train = X_train  # Lưu trữ dữ liệu training
        self.y_train = y_train  # Lưu trữ nhãn của tập training
        '''END'''
    # hàm predict
    # tương tự như X_train, X_test cũng phải lấy tần số từ rồi chuyển về mảng trước khi đưa vào dự đoán
    def predict(self, X_test):
        self.X_test = X_test

        _distance = cosine_distance(self.X_train, self.X_test) # tính khoảng cách tất cả các dòng trong tập test với tập train --> dict 
    
        # reset index y_train bắt đầu từ 1 - 15
        self.y_train.index = range(len(self.y_train))
        
        _distance_frame = pd.concat([pd.DataFrame(_distance), pd.DataFrame(self.y_train)], axis = 1)
        
        # B1: tạo frame với _distance và tạo frame với y_train (ban đầu ở dạng Series, tạo tên cột là target)
        # B2: hàm concat nối hai frame này lại với nhau (nối theo cột nên axis = 1)
        
        target_predict = dict() # tạo dict trống
        for i in range(1, len(self.X_test) + 1): # lặp qua các dòng trong X_test thông qua index, bắt đầu từ 1 vì ở trong hàm khoảng cách ta bắt đầu từ 1
            # lấy frame con chỉ hai cột i và target rồi sắp xếp cột i tăng dần, sau đó lấy k hàng đầu
            '''Code here'''
            temp_df = pd.DataFrame({'distance' : _distance[i], 'label' : self.y_train}) # # Tạo một DataFrame tạm thời từ khoảng cách tính được và nhãn của tập train
            # đếm tần số giá trị trong cột target rồi lấy phần tử có tần số cao nhất gán cho phần tử của X_test 
            '''Code here'''
            top_k_neightbors = temp_df.sort_values(by='distance', ascending=False).head(self.k) ## Sắp xếp theo khoảng cách tăng dần và lấy k điểm gần nhất
            top_label = top_k_neightbors['label'].mod()[0]
            # thêm key và giá trị gán tương ứng vào target_predict, Ví dụ: {1: [True], 2: [False]} 
            '''Code here'''
            target_predict[i] = top_label
            
        # return target_predict # trả lại dict đã dự đoán
    # tạo hàm score
    def score(self, y_test, y_pred):
        correct = 0 
        total = len(y_test)

        for i in range(total):
            if y_test.iloc[i] == y_pred[i + 1]:
                correct += 1
        return correct / total
        

### Demo bằng Education

data = loadCsv('ML_UD/lab4/lab3/Education.csv')
# loại bỏ các kí tự đặc biệt
data['Text'] = data['Text'].apply(lambda x: x.replace(',', ''))
data['Text'] = data['Text'].apply(lambda x: x.replace('.', ''))
data['Text'][1]

X_train, y_train, X_test, y_test = splitTrainTest(data, 0.25)

words_train_fre, bags = get_words_frequency(X_train)
# print(bags)
# print(len(bags))
words_train_fre = transform(X_test, bags)
knn = KNNText(k = 2)
knn.fit(words_train_fre.values, y_train)
print(X_test, X_train)
cosine_distance(int(X_train), int(X_test))
knn.predict(X_test)
print(len(X_train))
print(len(X_test))

words_train_fre, bags = get_words_frequency(X_train)
print(bags)
print(len(bags))
print(words_train_fre)

words_train_fre

words_test_fre = transform(X_test, bags)

words_test_fre

knn = KNNText(k = 2)

knn.fit(words_train_fre.values, y_train)
# knn.predict(X_test)
y_pred = knn.predict(words_test_fre.values)

accuracy = knn.score(y_test, y_pred)
print(f'Độ chính xác của mô hình: {accuracy * 100:.2f}%')


pred_ =  pd.DataFrame(pd.DataFrame(knn.predict(words_test_fre.values)).values.reshape(-1), columns = ['Predict'])
pred_.index = range(1, len(pred_) + 1)

y_test.index = range(1, len(y_test)+ 1)
y_test = y_test.to_frame(name = 'Actual')

result = pd.concat([pred_, y_test], axis = 1)
print(result)