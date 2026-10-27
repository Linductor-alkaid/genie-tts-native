// src/text/opencpop_table.cpp
//
// 本文件由 tools/g2p/generate_opencpop_table.py 生成，请勿手工修改。
//
// 数据来源：上游 Genie-TTS 随资源分发的 opencpop-strict.txt
// （拼音键 → "声母 韵母"，429 条，按键升序）。
// 上游用它把 to_initials/to_finals_tone3 的切分结果转成音素符号对
// （ChineseG2P._pinyin_to_opencpop_phones），是符号转换的唯一权威
// —— 翘舌 ir、ü 的 v、二声 EE 等特殊表示全部来自这张表。

#include "text/opencpop_table.hpp"

#include <mutex>

namespace genie::pinyin::detail {

namespace {

/// 惰性解析："拼音键\t声母 韵母" 行集 → 查找表。
/// 行尾的 \r 要去掉（Windows checkout 是 CRLF）。
const OpencpopLexicon& lexicon() {
    static const OpencpopLexicon kLexicon = [] {
        OpencpopLexicon table;
        std::string_view text = kOpencpopTable;
        while (!text.empty()) {
            const auto newline = text.find('\n');
            std::string_view line =
                text.substr(0, newline == std::string_view::npos ? text.size() : newline);
            if (!line.empty() && line.back() == '\r') {
                line.remove_suffix(1);
            }
            const auto tab = line.find('\t');
            if (tab != std::string_view::npos) {
                const auto key = line.substr(0, tab);
                const auto space = line.find(' ', tab + 1);
                if (space != std::string_view::npos) {
                    table.emplace(std::string(key),
                                  std::make_pair(std::string(line.substr(tab + 1, space - tab - 1)),
                                                 std::string(line.substr(space + 1))));
                }
            }
            if (newline == std::string_view::npos) {
                break;
            }
            text = text.substr(newline + 1);
        }
        return table;
    }();
    return kLexicon;
}

}  // namespace

const OpencpopLexicon& opencpop_lexicon() {
    return lexicon();
}

const char* const kOpencpopTable = R"OPENCPOP(
a	AA a
ai	AA ai
an	AA an
ang	AA ang
ao	AA ao
ba	b a
bai	b ai
ban	b an
bang	b ang
bao	b ao
bei	b ei
ben	b en
beng	b eng
bi	b i
bian	b ian
biao	b iao
bie	b ie
bin	b in
bing	b ing
bo	b o
bu	b u
ca	c a
cai	c ai
can	c an
cang	c ang
cao	c ao
ce	c e
cei	c ei
cen	c en
ceng	c eng
cha	ch a
chai	ch ai
chan	ch an
chang	ch ang
chao	ch ao
che	ch e
chen	ch en
cheng	ch eng
chi	ch ir
chong	ch ong
chou	ch ou
chu	ch u
chua	ch ua
chuai	ch uai
chuan	ch uan
chuang	ch uang
chui	ch ui
chun	ch un
chuo	ch uo
ci	c i0
cong	c ong
cou	c ou
cu	c u
cuan	c uan
cui	c ui
cun	c un
cuo	c uo
da	d a
dai	d ai
dan	d an
dang	d ang
dao	d ao
de	d e
dei	d ei
den	d en
deng	d eng
di	d i
dia	d ia
dian	d ian
diao	d iao
die	d ie
ding	d ing
diu	d iu
dong	d ong
dou	d ou
du	d u
duan	d uan
dui	d ui
dun	d un
duo	d uo
e	EE e
ei	EE ei
en	EE en
eng	EE eng
er	EE er
fa	f a
fan	f an
fang	f ang
fei	f ei
fen	f en
feng	f eng
fo	f o
fou	f ou
fu	f u
ga	g a
gai	g ai
gan	g an
gang	g ang
gao	g ao
ge	g e
gei	g ei
gen	g en
geng	g eng
gong	g ong
gou	g ou
gu	g u
gua	g ua
guai	g uai
guan	g uan
guang	g uang
gui	g ui
gun	g un
guo	g uo
ha	h a
hai	h ai
han	h an
hang	h ang
hao	h ao
he	h e
hei	h ei
hen	h en
heng	h eng
hong	h ong
hou	h ou
hu	h u
hua	h ua
huai	h uai
huan	h uan
huang	h uang
hui	h ui
hun	h un
huo	h uo
ji	j i
jia	j ia
jian	j ian
jiang	j iang
jiao	j iao
jie	j ie
jin	j in
jing	j ing
jiong	j iong
jiu	j iu
ju	j v
juan	j van
jue	j ve
jun	j vn
jv	j v
jvan	j van
jve	j ve
jvn	j vn
ka	k a
kai	k ai
kan	k an
kang	k ang
kao	k ao
ke	k e
kei	k ei
ken	k en
keng	k eng
kong	k ong
kou	k ou
ku	k u
kua	k ua
kuai	k uai
kuan	k uan
kuang	k uang
kui	k ui
kun	k un
kuo	k uo
la	l a
lai	l ai
lan	l an
lang	l ang
lao	l ao
le	l e
lei	l ei
leng	l eng
li	l i
lia	l ia
lian	l ian
liang	l iang
liao	l iao
lie	l ie
lin	l in
ling	l ing
liu	l iu
lo	l o
long	l ong
lou	l ou
lu	l u
luan	l uan
lun	l un
luo	l uo
lv	l v
lve	l ve
ma	m a
mai	m ai
man	m an
mang	m ang
mao	m ao
me	m e
mei	m ei
men	m en
meng	m eng
mi	m i
mian	m ian
miao	m iao
mie	m ie
min	m in
ming	m ing
miu	m iu
mo	m o
mou	m ou
mu	m u
na	n a
nai	n ai
nan	n an
nang	n ang
nao	n ao
ne	n e
nei	n ei
nen	n en
neng	n eng
ni	n i
nian	n ian
niang	n iang
niao	n iao
nie	n ie
nin	n in
ning	n ing
niu	n iu
nong	n ong
nou	n ou
nu	n u
nuan	n uan
nun	n un
nuo	n uo
nv	n v
nve	n ve
o	OO o
ou	OO ou
pa	p a
pai	p ai
pan	p an
pang	p ang
pao	p ao
pei	p ei
pen	p en
peng	p eng
pi	p i
pian	p ian
piao	p iao
pie	p ie
pin	p in
ping	p ing
po	p o
pou	p ou
pu	p u
qi	q i
qia	q ia
qian	q ian
qiang	q iang
qiao	q iao
qie	q ie
qin	q in
qing	q ing
qiong	q iong
qiu	q iu
qu	q v
quan	q van
que	q ve
qun	q vn
qv	q v
qvan	q van
qve	q ve
qvn	q vn
ran	r an
rang	r ang
rao	r ao
re	r e
ren	r en
reng	r eng
ri	r ir
rong	r ong
rou	r ou
ru	r u
rua	r ua
ruan	r uan
rui	r ui
run	r un
ruo	r uo
sa	s a
sai	s ai
san	s an
sang	s ang
sao	s ao
se	s e
sen	s en
seng	s eng
sha	sh a
shai	sh ai
shan	sh an
shang	sh ang
shao	sh ao
she	sh e
shei	sh ei
shen	sh en
sheng	sh eng
shi	sh ir
shou	sh ou
shu	sh u
shua	sh ua
shuai	sh uai
shuan	sh uan
shuang	sh uang
shui	sh ui
shun	sh un
shuo	sh uo
si	s i0
song	s ong
sou	s ou
su	s u
suan	s uan
sui	s ui
sun	s un
suo	s uo
ta	t a
tai	t ai
tan	t an
tang	t ang
tao	t ao
te	t e
tei	t ei
teng	t eng
ti	t i
tian	t ian
tiao	t iao
tie	t ie
ting	t ing
tong	t ong
tou	t ou
tu	t u
tuan	t uan
tui	t ui
tun	t un
tuo	t uo
wa	w a
wai	w ai
wan	w an
wang	w ang
wei	w ei
wen	w en
weng	w eng
wo	w o
wu	w u
xi	x i
xia	x ia
xian	x ian
xiang	x iang
xiao	x iao
xie	x ie
xin	x in
xing	x ing
xiong	x iong
xiu	x iu
xu	x v
xuan	x van
xue	x ve
xun	x vn
xv	x v
xvan	x van
xve	x ve
xvn	x vn
ya	y a
yan	y En
yang	y ang
yao	y ao
ye	y E
yi	y i
yin	y in
ying	y ing
yo	y o
yong	y ong
you	y ou
yu	y v
yuan	y van
yue	y ve
yun	y vn
yv	y v
yvan	y van
yve	y ve
yvn	y vn
za	z a
zai	z ai
zan	z an
zang	z ang
zao	z ao
ze	z e
zei	z ei
zen	z en
zeng	z eng
zha	zh a
zhai	zh ai
zhan	zh an
zhang	zh ang
zhao	zh ao
zhe	zh e
zhei	zh ei
zhen	zh en
zheng	zh eng
zhi	zh ir
zhong	zh ong
zhou	zh ou
zhu	zh u
zhua	zh ua
zhuai	zh uai
zhuan	zh uan
zhuang	zh uang
zhui	zh ui
zhun	zh un
zhuo	zh uo
zi	z i0
zong	z ong
zou	z ou
zu	z u
zuan	z uan
zui	z ui
zun	z un
zuo	z uo
)OPENCPOP";

}  // namespace genie::pinyin::detail
