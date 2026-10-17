import { supportsStrictTools } from './model-strictness';

describe('supportsStrictTools', () => {
    it('enables strict only for Gemini', () => {
        expect(supportsStrictTools('gemini-3.7-flash')).toBe(true);
        expect(supportsStrictTools('gemini_2_flash')).toBe(true);
        expect(supportsStrictTools('gpt-5.6-luna')).toBe(false);
        expect(supportsStrictTools('claude-opus-4-8')).toBe(false);
        expect(supportsStrictTools(undefined)).toBe(false);
    });
});
