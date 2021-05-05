import { RGBAToHexA } from "./utils.js";

function setProgress(data) {
    $('#formality-progress')
        .progress({
            duration: 200,
            percent: parseInt(data.formality.prob * 100),
            text: {
                active: 'Formality : {percent} %'
            }
        })
        ;
    $('#emo-progress')
        .progress({
            duration: 200,
            percent: parseInt(data.jokes.prob * 100),
            text: {
                active: 'Emotion : {percent} %'
            }
        })
        ;
}

function displayModal(data) {
    let modalHTML = '';

    let originalText = "<div class='row middle-xs'>" +
        "<p class='modal-text'><span class='ui label'>Original Text</span><span class='original'>" + data.input.text + "</span></p>" +
        "<div class='ui teal image label'>70%<div class='detail'>Formality</div></div>" +
        "<div class='ui yellow image label'>2%<div class='detail'>Emotion</div></div>" +
        "</div>";
    let suggestionText = '<h4 class="ui horizontal divider header"><i class="lightbulb icon"></i></h4>';
    for (var i = 0; i < data.suggestions.length; i++) {
        suggestionText += "<div class='row middle-xs suggestion-item' data-suggestion-id='" + i + "'>" +
            "<p class='modal-text'><span class='ui label'>Suggestion " + (i + 1) + "</span><span class='suggestion'>" + data.suggestions[i].text + "</span></p>" +
            "<div class='ui teal image label'>70%<div class='detail'>Formality</div></div>" +
            "<div class='ui yellow image label'>2%<div class='detail'>Emotion</div></div>" +
            "</div>";
    }
    modalHTML = originalText + suggestionText;
    // console.log(modalHTML);
    $('#transfer-content').html(modalHTML);
}

function setSliders(data) {
    // $('#formality-slider').slider('set value', data.formality.probBucket);
    // $('#emo-slider').slider('set value', data.jokes.probBucket);

    $('#formality-slider').slider('set value', 2);
    $('#emo-slider').slider('set value', 0);

}

function displayHeatmap(data) {
    let output_txt = ''
    let recent_end = 0
    for (let i = 0; i < data.tokens.length; i++) {
        let currToken = data.tokens[i]
        if (recent_end != currToken.start) {
            output_txt += " ";
        }
        output_txt += "<span style='background : " + RGBAToHexA(255, 0, 0, currToken.attention / 20) + "'>" + currToken.text + "</span>"
    }
    $('#preview').html(output_txt);
}

// function displayJointHeatmap(data, dropdownSelection) {
//     let output_txt = ''
//     let recent_end = 0
//     let query = null;
//     if (dropdownSelection === 'formality') {
//         query = 'formality'
//     }
//     else if (dropdownSelection === 'emo') {
//         query = 'jokes'
//     }

//     if (query != null) {
//         for (let i = 0; i < data.tokens.length; i++) {
//             let currToken = data.tokens[i]
//             if (recent_end != currToken.start) {
//                 output_txt += " ";
//                 recent_end += 1;
//             }
//             output_txt += "<span style='background : " + RGBAToHexA(255, 0, 0, data[query]['salience'][i] / 2) + "'>" + currToken.text + "</span>"
//             recent_end += currToken.text.length
//         }
//         $('#preview').html(output_txt);
//     }
//     else {
//         $('#preview').html(data.input);
//     }

// }

function displayJointHeatmap(data, dropdownSelection, quillEditor) {
    let output_txt = []
    let recent_end = 0
    let query = null;
    if (dropdownSelection === 'formality') {
        query = 'formality'
    }
    else if (dropdownSelection === 'emo') {
        query = 'jokes'
    }

    if (query != null) {
        for (let i = 0; i < data.tokens.length; i++) {
            let currToken = data.tokens[i]
            if (recent_end != currToken.start) {
                output_txt.push({ insert: ' ' });
                recent_end += 1;
            }
            output_txt.push({ insert: currToken.text, attributes: { background: RGBAToHexA(255, 0, 0, data[query]['salience'][i] / 2) } })
            recent_end += currToken.text.length
        }
        // console.log(output_txt);
        quillEditor.setContents(output_txt);
    }
    else {
        quillEditor.setContents([{ insert: data.input }]);
    }

}


export { displayHeatmap, displayJointHeatmap, setProgress, setSliders, displayModal };