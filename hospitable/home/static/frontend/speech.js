window.SpeechRecognition = window.webkitSpeechRecognition || window.SpeechRecognition;
const synth = window.speechSynthesis;
const recognition = new SpeechRecognition();

const icon = document.querySelector('i.fa.fa-microphone')
let paragraph = document.createElement('p');
let container = document.querySelector('.text-box');

const sound = document.querySelector('.sound');

let raiseHand = document.getElementById('raise-hand');
let muteButton = document.getElementById('mute-unmute');
let okButton = document.getElementById('ok');
let endButton = document.getElementById('endbutton');


icon.addEventListener('click', () => {
    sound.play();
    dictate();
});

const dictate = () => {
    recognition.start();
    recognition.onresult = (event) => {
        const speechToText = event.results[0][0].transcript;

        container.appendChild(paragraph);

        paragraph.textContent = speechToText;

        if (event.results[0].isFinal) {

            if (speechToText == 'mute') {
                muteButton.innerHTML = "Unmute";
            };

            if (speechToText == 'Unmute') {
                muteButton.innerHTML = "Mute";
            };
            if (speechToText == 'raise hand') {
                raiseHand.innerHTML = "Lower Hand";
            };

            if (speechToText == 'lower hand') {
                raiseHand.innerHTML = "Raise Hand";
            };

            if (speechToText == 'I agree') {
                okButton.innerHTML = "ok";
            };

            if (speechToText === 'ok') {
                okButton.innerHTML = "I agree";
            };

            if (speechToText === 'leave meeting') {

                setTimeout(() => { alert("Meeting has ended"); }, 2000);
            };

            if (speechToText.includes('what is today\'s date')) {
                alert("Hello");
                speak(getDate);
            };

            if (speechToText.includes('what is the weather in')) {
                getTheWeather(speechToText);
            };
        }
    }
}


const speak = (action) => {
    utterThis = new SpeechSynthesisUtterance(action());
    synth.speak(utterThis);
};
