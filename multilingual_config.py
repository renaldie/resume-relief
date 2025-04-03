# Configuration for multilingual text processing

# Chinese stopwords - common words to filter out
CHINESE_STOPWORDS = [
    '的', '了', '和', '是', '就', '都', '而', '及', '與', '著',
    '或', '一個', '沒有', '我們', '你們', '他們', '她們', '自己',
    '之', '與', '質', '所以', '因此', '以及', '也', '但是', '因為',
    '如果', '那麼', '不過', '為了', '雖然', '還是', '只有', '需要',
    '這個', '那個', '一些', '可以', '已經', '不', '在', '對', '上',
    '下', '有', '會', '時', '能', '靠', '怎麼', '什麼', '誰', '得',
    '將', '向', '等', '從', '到', '於', '被', '它', '他', '她', '我',
    '你', '您', '每', '年', '經驗', '以上', '崗位', '職位', '工作'
]

# Job-specific Chinese terms (to be preserved during processing)
CHINESE_JOB_TERMS = {
    '全職': 'full_time',
    '兼職': 'part_time',
    '實習': 'internship',
    '經驗': 'experience',
    '學歷': 'education',
    '本科': 'bachelor_degree',
    '碩士': 'masters_degree',
    '博士': 'phd',
    '職責': 'responsibilities',
    '要求': 'requirements',
    '技能': 'skills',
    '資格': 'qualifications',
    '薪資': 'salary',
    '福利': 'benefits',
    '工作環境': 'work_environment',
    '團隊': 'team',
    '公司': 'company',
    '管理': 'management',
    '領導': 'leadership',
    '溝通': 'communication',
    '分析': 'analysis',
    '設計': 'design',
    '開發': 'development',
    '測試': 'testing',
    '研究': 'research',
    '銷售': 'sales',
    '市場': 'marketing',
    '客戶': 'customer',
    '財務': 'finance',
    '人力資源': 'human_resources',
    '操作': 'operations',
    '項目': 'project',
    '產品': 'product'
}

# Dictionary mapping of Chinese degree terms to English
DEGREE_MAPPING = {
    '本科': 'bachelor',
    '學士': 'bachelor',
    '碩士': 'master',
    '博士': 'phd',
    '研究生': 'graduate',
    '大專': 'associate_degree',
    '高中': 'high_school',
    '職業學校': 'vocational_school',
    '中專': 'technical_school',
    '大學': 'university',
    '學院': 'college'
}

# Common Chinese tech skills
TECH_SKILLS_MAPPING = {
    '程序設計': 'programming',
    '編程': 'coding',
    '數據庫': 'database',
    '算法': 'algorithms',
    '人工智能': 'artificial_intelligence',
    '機器學習': 'machine_learning',
    '深度學習': 'deep_learning',
    '數據分析': 'data_analysis',
    '數據挖掘': 'data_mining',
    '網絡安全': 'cybersecurity',
    '雲計算': 'cloud_computing',
    '大數據': 'big_data',
    '前端': 'frontend',
    '後端': 'backend',
    '全棧': 'fullstack',
    '移動開發': 'mobile_development',
    '應用程序': 'application',
    '系統架構': 'system_architecture',
    '軟件工程': 'software_engineering',
    '項目管理': 'project_management',
    '敏捷開發': 'agile_development',
    '測試': 'testing',
    '質量保證': 'quality_assurance',
    '用戶界面': 'user_interface',
    '用戶體驗': 'user_experience'
}

# Function to load Chinese spaCy model
def load_chinese_nlp():
    try:
        import spacy
        # Try to load Chinese model
        try:
            nlp_zh = spacy.load("zh_core_web_sm")
            return nlp_zh
        except:
            print("Chinese spaCy model not found. Install with: python -m spacy download zh_core_web_sm")
            return None
    except ImportError:
        print("spaCy not installed. Install with: pip install spacy")
        return None