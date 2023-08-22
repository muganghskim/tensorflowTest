// helpers.ts
const alphabet = "abcdefghijklmnopqrstuvwxyz".split("");

// 문자열을 원-핫 인코딩으로 변환
export const oneHotEncode = (text: string, maxLength: number) => {
  const encodedText = Array(maxLength)
    .fill(null)
    .map(() => Array(26).fill(0));

  for (let i = 0; i < text.length && i < maxLength; i++) {
    const char = text[i];
    const index = alphabet.indexOf(char.toLowerCase());
    if (index !== -1) {
      encodedText[i][index] = 1;
    }
  }
  return encodedText;
};

// 입력, 출력 텍스트를 전처리
export const preprocessText = (text: string): string => {
  return text;
};

// 결과를 텍스트로 변환
export const sample = (predictions: number[], maxLength: number) => {
  let result = "";
  for (let i = 0; i < predictions.length; i += alphabet.length) {
    const currentPrediction = predictions.slice(i, i + alphabet.length);
    const maxIndex = currentPrediction.indexOf(Math.max(...currentPrediction));
    result += alphabet[maxIndex];
  }
  return result;
};
