import pandas as pd
from common import save_file

#----------------------------------------------------------------------

http_counter_list = []

for i in range(df.shape[0]):
    http_counter = 0
    z = nltk.word_tokenize(df['posts'][i])
    for j in range(len(z)):
        if 'http' in z[j]:
            http_counter += 1
    http_counter_list.append(http_counter)
#----------------------------------------------------------------------

website_header_list = []

for i in range(df.shape[0]):
    temporary_header_list = []
    post = df['posts'][i]
    link_list = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',post)
    if link_list:
        for link in link_list:
            try:
                read_url = request.urlopen(link).read().decode('utf-8')
            except:
                pass
            title_of_link = re.findall(r"(?<=<title>).*(?=<\/title>)",read_url)
            temporary_header_list.append(''.join(title_of_link))
    else:
        temporary_header_list.append("")
    #string_websites = "|||".join(temporary_header_list)
    website_header_list.append("|||".join(temporary_header_list))
#----------------------------------------------------------------------

df = pd.read_excel("./features/kaggle/http_verb_adjective_weblink_features2.xlsx")

word_lens = df["processed_post"].str.split(" ").str.len()

df_new = pd.DataFrame(df["http_counter"]/word_lens, columns=["http_frac"])
df_new["idx"] = df["idx"]

df_new.loc[df_new["http_frac"].isnull(), "http_frac"] = 0

save_file(df_new, "./features/kaggle/http_df.pickle")
