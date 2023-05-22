window.onload = function() {
    const video = document.getElementById("video");
    const shutter = document.getElementById("shutter");
    
    navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false
    }).then(stream => {
        video.srcObject = stream;
        video.play();
    }).catch(e => {
        console.log(e);
    });
    
    // function onClickVideo() 
    document.getElementById('shutter').onclick = function() {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const context = canvas.getContext('2d');
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const imageDataURL = canvas.toDataURL('image/png');
    
        // Ajaxリクエストで写真データを送信
        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/upload_photo', true);
        xhr.setRequestHeader('Content-Type', 'application/json');
        xhr.onreadystatechange = function () {
            if (xhr.readyState === XMLHttpRequest.DONE && xhr.status === 200) {
                console.log('写真が送信されました');
            }
        };
        xhr.send(JSON.stringify({ photo: imageDataURL }));
    
        const element = document.getElementById('indicator');
    
        element.textContent = 'これはコカ・コーラです';
    
        speak(); // テスト用 クリックしたら読み上げされるように呼び出し
    };
    
    function speak(){
        var speak   = new SpeechSynthesisUtterance();
        speak.text  = document.getElementById('indicator').textContent;
        speak.rate  = 1; // 読み上げ速度 0.1-10 初期値:1 (倍速なら2, 半分の倍速なら0.5, )
        speak.pitch = 0; // 声の高さ 0-2 初期値:1(0で女性の声) 
        speak.lang  = 'ja-JP'; //(日本語:ja-JP, アメリカ英語:en-US, イギリス英語:en-GB, 中国語:zh-CN, 韓国語:ko-KR)
      
        sleep(2000);
        speechSynthesis.speak(speak);
      
    };
      
    function sleep(time){
        var date_1 = new Date().getTime();
        var date_2 = new Date().getTime();
        while (date_2 < date_1 + time){
            date_2 = new Date().getTime();
        }
        return;
    };
    
}

