
ref_anhquan_neutral = '/mnt/nfs-shared/kilm/users/lampv/prepare_data_tts/segments/anhquan/disco trưa 29 7_0.wav'
ref_anhquan_soft = '/mnt/nfs-shared/kilm/users/lampv/prepare_data_tts/segments/anhquan/Podcast 23_7_Audio_3.wav'
ref_anhquan_energetic = '/mnt/nfs-shared/kilm/users/lampv/prepare_data_tts/segments/anhquan/pbg-26-7_0.wav'

ref_numienbac = '/mnt/nfs-shared/kilm/users/lampv/prepare_data_tts_old/datasets/numienbac/wavs_24k/day23-001-000394-001259.wav'
ref_nammienbac = '/mnt/nfs-shared/kilm/users/lampv/prepare_data_tts_old/datasets/nammienbac/wavs_24k/kinh-doanh-00131-000787-001623.wav'
ref_numiennam = '/mnt/nfs-shared/kilm/users/lampv/prepare_data_tts_old/datasets/numiennam/wavs_24k/bat-dong-san-00000-002837-003533.wav'
ref_nammiennam = '/mnt/nfs-shared/kilm/users/lampv/prepare_data_tts_old/datasets/nammiennam/wavs_24k/bat-dong-san-00043-018155-019165.wav'

ref_huonglinh_vov = '/mnt/nfs-shared/kilm/users/lampv/prepare_data_tts/segments/huonglinh-vov/1_1.wav'
ref_huonglinh_thoisu = '/mnt/nfs-shared/kilm/users/lampv/prepare_data_tts/segments/huonglinh-thoisu/qx4-DL6rELw_0.wav'

ref_vietha_vov = '/mnt/nfs-shared/kilm/users/lampv/prepare_data_tts/segments/vietha-vov/P1_7.wav'
ref_vietha_thoisu = '/mnt/nfs-shared/kilm/users/lampv/prepare_data_tts/segments/vietha-thoisu/1_0.wav'

ref_conghieu_bongda = '/mnt/nfs-shared/kilm/users/lampv/prepare_data_tts/segments/conghieu-bongda/3.3_1.wav'
ref_kplus1 = '/mnt/nfs-shared/kilm/users/lampv/prepare_data_tts/segments/kplus_1/6-hFu31IH9w_3.wav'
ref_cambongda = '/mnt/nfs-shared/kilm/users/lampv/prepare_data_tts/segments/cambongda/0CNdsfLpYcs_6.wav'
ref_tienphong = '/mnt/nfs-shared/kilm/users/lampv/prepare_data_tts/segments/tienphong/4mFfaziSRKA_534.0_0.wav'

ref_hoangthuy_thoisu = '/mnt/nfs-shared/kilm/users/lampv/prepare_data_tts/segments/hoangthuy-thoisu/-FjzBhj4yQY_t1013_03.wav'

ref_nhungoc = '/mnt/nfs-shared/kilm/users/lampv/prepare_data_tts/segments/nhungoc/1_1.wav'
ref_tuoitre_male = '/mnt/nfs-shared/kilm/users/lampv/prepare_data_tts/segments/tuoitre_male/1boFW-UPBxE_1.0_0.wav'
ref_tuoitre_female = '/mnt/nfs-shared/kilm/users/lampv/prepare_data_tts/segments/tuoitre_female/_PsJ_E63WQs_1.0_1.wav'
ref_thuvienphapluat = '/mnt/nfs-shared/kilm/users/lampv/prepare_data_tts/segments/thuvienphapluat/6KM5eQTGul8_0.wav'


ref_thuylinh = '/mnt/nfs-shared/kilm/users/lampv/prepare_data_tts/segments/thuylinh/1_2.wav'
ref_hieubiethon = '/mnt/nfs-shared/kilm/users/lampv/prepare_data_tts/segments/hieubiethon/66ENgFFXZnI_4.0_1.wav'


def prepare_anhquan(style, model):
    if style == 'neutral':
        ref_audio = ref_anhquan_neutral
        text_demo = 'chỉ trong vòng sáu tháng qua , đô-nan trăm đã kết_thúc nhiều cuộc_chiến , tự_hào nhận mình là tổng_thống hòa_bình , sau khi chính_thức công_bố việc hai quốc_gia thái_lan và cam-pu-chia đạt được thỏa_thuận ngừng bắn'
    elif style == 'energetic':
        ref_audio = ref_anhquan_energetic
        text_demo = 'quỷ đỏ đang thực_sự cần một thủ_môn chất_lượng , khi an-<đ>-rê ô-na-na thiếu ổn_định , thậm chí , ngôi_sao người ca-mơ-run bất_ngờ dính chấn_thương , qua đó bỏ_lỡ giai_đoạn tiền mùa_giải của quỷ đỏ thành man-chét-<x>-tơ'
    elif style == 'soft':
        ref_audio = ref_anhquan_soft
        text_demo = 'giờ đây , chỉ còn lại sự tĩnh_lặng đến xót_xa . sự ra đi đột_ngột của em trân ~ không chỉ cướp đi một tâm_hồn trong_trẻo ~ mà còn kéo theo những vết_thương không bao giờ lành trong trái_tim của những người ở lại'

    ref_embedding = model.compute_ref_emb(ref_audio)
    
    return ref_embedding, text_demo


def prepare_phuonglinh(style, model):
    if style == 'news_host':
        ref_embedding = model.compute_ref_emb_mix(ref_huonglinh_vov, ref_speaker_audio_path_2=ref_thuylinh)
        text_demo = 'và trong_sáng ngày hôm_nay có những thông_tin nào đáng chú_ý của thị_trường tài_chính kinh_doanh , chúng ta sẽ cùng gặp_gỡ biên_tập viên minh tú để cập_nhật ngay bây_giờ . xin chào anh tú'
    if style == 'news':
        ref_embedding = model.compute_ref_emb_mix(ref_huonglinh_thoisu, ref_hieubiethon, ratio=3)
        text_demo = 'nhân dịp ông đô-nan trăm chính_thức nhậm_chức tổng_thống thứ bốn_mươi_bảy của hợp_chủng_quốc hoa_kỳ , ngày hai_mươi_mốt tháng một theo giờ hà_nội , tổng bí_thư tô lâm , chủ_tịch nước lương cường và thủ_tướng chính_phủ phạm minh chính đã gửi thư chúc_mừng'

    return ref_embedding, text_demo


def prepare_thanhha(style, model):
    if style == 'news_host':
        ref_embedding = model.compute_ref_emb_mix(ref_vietha_thoisu, ref_vietha_vov, ref_thuylinh, ref_vietha_vov)
        text_demo = 'quý thính_giả thân_mến , quyết_định tăng lãi_suất cơ_bản của ngân_hàng trung_ương sẽ tác_động như_thế_nào đến thị_trường chứng_khoán và các khoản vay tiêu_dùng ? đây sẽ là nội_dung chính được chúng_tôi phân_tích trong chuyên_mục tiêu_điểm kinh_tế'
    elif style == 'news':
        ref_embedding = model.compute_ref_emb(ref_vietha_thoisu)
        text_demo = 'tổng bí_thư tô lâm cũng nhấn_mạnh vai_trò của hiệp_hội dữ_liệu quốc_gia trong việc thúc_đẩy phát_triển và khai_thác dữ_liệu , đồng_thời đề_xuất bảy nhiệm_vụ trọng_tâm cho hiệp_hội để thực_hiện sứ_mệnh này'
    
    return ref_embedding, text_demo


def prepare_thuydung(style, model):
    if style == 'news':
        ref_embedding = model.compute_ref_emb_mix(ref_numienbac, ref_hoangthuy_thoisu, ratio=3)
        text_demo = 'ngoài ra , cũng không loại_trừ khả_năng nga đang tích_lũy tên_lửa cho những cuộc tấn_công quy_mô lớn hơn trong thời_gian tới . nhà phân_tích mô-si-an-cô nhận_định rằng , về mặt quân_sự , nga nhiều khả_năng đã hoàn_tất hầu hết các giai_đoạn chuẩn_bị'
    
    return ref_embedding, text_demo


def prepare_tronghieu(style, model):
    if style == 'news':
        ref_embedding = model.compute_ref_emb_mix(ref_conghieu_bongda, ref_tienphong,  ref_prosody_audio_path_2=ref_cambongda, ratio=3)
        text_demo = 'các quan_chức trong chính_quyền ông trăm đã liên_tục hạ_thấp mức_độ nghiêm_trọng của tình_trạng biến_động kinh_tế . cố_vấn cấp cao về thương_mại và sản_xuất của tổng_thống trăm , ông pi-tơ na-va-rô kêu_gọi các nhà đầu_tư mỹ ngồi yên và đừng hoảng_sợ'
    elif style == 'sport':
        ref_embedding = model.compute_ref_emb_mix(ref_conghieu_bongda, ref_kplus1, ref_prosody_audio_path_2=ref_cambongda, ratio=3)
        text_demo = 'tâm_điểm của thể_thao thế_giới cuối tuần này sẽ là cuộc đối_đầu kinh_điển giữa hai gã khổng_lồ của bóng_đá châu_âu . không chỉ là cuộc_chiến vì danh_dự , trận_đấu này còn có thể quyết_định ngôi vương của cả mùa_giải '

    return ref_embedding, text_demo


def prepare_minhtu(style, model):
    if style == 'news':
        text_demo = 'vâng thưa quý_vị , hiện_nay nhiều ngân_hàng tại việt_nam đang tăng lãi_suất huy_động để đối_phó với áp_lực . xu_hướng tăng lãi_suất này ảnh_hưởng đến cả các ngân_hàng lớn_nhỏ và các ngân_hàng thương_mại'
        ref_embedding = model.compute_ref_emb_mix(ref_nammienbac, ref_nhungoc, ref_prosody_audio_path_2=ref_nhungoc)

    return ref_embedding, text_demo


def prepare_maiyen(style, model):
    if style == 'news':
        text_demo = 'cựu thủ_tướng thái_lan xuất_hiện tại văn_phòng ban kiểm_soát ma_túy , dập tắt tin_đồn ông có kế_hoạch trốn khỏi thái_lan , để tránh chịu hậu_quả pháp_lý liên_quan đến thời_gian nằm viện gây tranh_cãi'
        ref_embedding = model.compute_ref_emb_mix(ref_numiennam, ref_tuoitre_female, ref_speaker_audio_path_2=ref_numiennam)

    return ref_embedding, text_demo


def prepare_vietkhuong(style, model):
    if style == 'news':
        ref_embedding_1 = model.compute_ref_emb_mix(ref_nammiennam, ref_tuoitre_male)
        ref_embedding_2 = model.compute_ref_emb_mix(ref_nammiennam, ref_thuvienphapluat)
        ref_embedding = (ref_embedding_1 + ref_embedding_2) / 2
        text_demo = 'ngày hai mươi mốt tháng năm , lực_lượng chức_năng phường tây thạnh , quận tân_phú , thành_phố hồ_chí_minh đang làm rõ sự_việc một chiếc xe_hơi bảy chỗ mang biển_số tỉnh tây_ninh đậu bên đường phạm ngọc thảo thuộc phường tây thạnh suốt thời_gian dài'
        
    return ref_embedding, text_demo



def prepare_voices(voice_name, style, model):
    if voice_name == 'anhquan':
        ref_embedding, text_demo = prepare_anhquan(style, model)
    elif voice_name == 'phuonglinh':
        ref_embedding, text_demo = prepare_phuonglinh(style, model)
    elif voice_name == 'thanhha':
        ref_embedding, text_demo = prepare_thanhha(style, model)
    elif voice_name == 'thuydung':
        ref_embedding, text_demo = prepare_thuydung(style, model)
    elif voice_name == 'minhtu':
        ref_embedding, text_demo = prepare_minhtu(style, model)
    elif voice_name == 'tronghieu':
        ref_embedding, text_demo = prepare_tronghieu(style, model)
    elif voice_name == 'maiyen':
        ref_embedding, text_demo = prepare_maiyen(style, model)
    elif voice_name == 'vietkhuong':
        ref_embedding, text_demo = prepare_vietkhuong(style, model)

    return ref_embedding, text_demo
